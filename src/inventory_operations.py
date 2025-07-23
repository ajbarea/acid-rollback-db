"""Inventory-specific operations for supermarket inventory management."""

import copy
import time
from typing import Dict, Any, Optional


class InventoryOperations:
    """
    Domain-specific inventory operations for the Regional Supermarket system.

    This class provides high-level inventory management methods that use the underlying ACID database for fault prevention and recovery.
    """

    def __init__(self, database):
        """
        Initialize inventory operations with a database instance.

        Args:
            database: AcidRollbackDB instance
        """
        self.db = database

    def update_stock_level(
        self, txn_id: str, sku: str, quantity_change: int, reason: str = "sale"
    ) -> bool:
        """
        Update stock level for a product within a transaction.

        Args:
            txn_id: Transaction ID
            sku: Product SKU
            quantity_change: Change in quantity (negative for sales, positive for restocking)
            reason: Reason for the change (sale, return, adjustment, restock)

        Returns:
            bool: True if successful, False if insufficient stock

        Raises:
            ValueError: If transaction is invalid or product not found
        """
        with self.db.lock:
            if not self.db.transaction_manager.is_transaction_active(txn_id):
                raise ValueError(f"Transaction {txn_id} not found or not active")

            current_item = self.db.data.get(sku)
            if not current_item:
                raise ValueError(f"Product {sku} not found in inventory")

            new_quantity = current_item["quantity"] + quantity_change

            # Prevent negative inventory
            if new_quantity < 0:
                raise ValueError(
                    f"Insufficient stock for {sku}. Available: {current_item['quantity']}, Requested: {abs(quantity_change)}"
                )

            # Create updated inventory record
            updated_item = copy.deepcopy(current_item)
            updated_item["quantity"] = new_quantity
            updated_item["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            updated_item["last_operation"] = reason

            # Use existing put method to maintain transaction consistency
            self.db.put(txn_id, sku, updated_item)

            print(
                f"Stock updated for {sku}: {current_item['quantity']} -> {new_quantity} ({reason})"
            )
            return True

    def reserve_inventory(self, txn_id: str, sku: str, quantity: int) -> bool:
        """
        Reserve inventory for a pending transaction (e.g., online order).

        Args:
            txn_id: Transaction ID
            sku: Product SKU
            quantity: Quantity to reserve

        Returns:
            bool: True if successful

        Raises:
            ValueError: If insufficient available stock
        """
        with self.db.lock:
            if not self.db.transaction_manager.is_transaction_active(txn_id):
                raise ValueError(f"Transaction {txn_id} not found or not active")

            current_item = self.db.data.get(sku)
            if not current_item:
                raise ValueError(f"Product {sku} not found in inventory")

            available_stock = current_item["quantity"] - current_item.get("reserved", 0)

            if available_stock < quantity:
                raise ValueError(
                    f"Insufficient available stock for {sku}. Available: {available_stock}, Requested: {quantity}"
                )

            # Update reservation
            updated_item = copy.deepcopy(current_item)
            updated_item["reserved"] = updated_item.get("reserved", 0) + quantity
            updated_item["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            updated_item["last_operation"] = "reservation"

            self.db.put(txn_id, sku, updated_item)

            print(
                f"Reserved {quantity} units of {sku}. Total reserved: {updated_item['reserved']}"
            )
            return True

    def process_return(self, txn_id: str, sku: str, quantity: int) -> bool:
        """
        Process a product return, increasing inventory.

        Args:
            txn_id: Transaction ID
            sku: Product SKU
            quantity: Quantity being returned

        Returns:
            bool: True if successful
        """
        return self.update_stock_level(txn_id, sku, quantity, "return")

    def adjust_inventory(
        self, txn_id: str, sku: str, new_quantity: int, reason: str = "audit"
    ) -> bool:
        """
        Adjust inventory to a specific quantity (for audits, corrections).

        Args:
            txn_id: Transaction ID
            sku: Product SKU
            new_quantity: New absolute quantity
            reason: Reason for adjustment

        Returns:
            bool: True if successful
        """
        with self.db.lock:
            if not self.db.transaction_manager.is_transaction_active(txn_id):
                raise ValueError(f"Transaction {txn_id} not found or not active")

            current_item = self.db.data.get(sku)
            if not current_item:
                raise ValueError(f"Product {sku} not found in inventory")

            old_quantity = current_item["quantity"]
            quantity_change = new_quantity - old_quantity

            # Create updated inventory record
            updated_item = copy.deepcopy(current_item)
            updated_item["quantity"] = new_quantity
            updated_item["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            updated_item["last_operation"] = reason
            updated_item["adjustment_note"] = (
                f"Adjusted from {old_quantity} to {new_quantity}"
            )

            self.db.put(txn_id, sku, updated_item)

            print(
                f"Inventory adjusted for {sku}: {old_quantity} -> {new_quantity} ({reason})"
            )
            return True

    def get_product_info(
        self, sku: str, txn_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed product information including availability.

        Args:
            sku: Product SKU
            txn_id: Optional transaction ID

        Returns:
            dict: Product information with calculated availability
        """
        product = self.db.get(sku, txn_id)
        if not product:
            return None

        # Calculate available stock (total - reserved)
        available = product["quantity"] - product.get("reserved", 0)

        # Add calculated fields
        product_info = copy.deepcopy(product)
        product_info["available"] = available
        product_info["is_in_stock"] = available > 0

        return product_info

    def check_stock_availability(
        self, sku: str, requested_quantity: int, txn_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if sufficient stock is available for a requested quantity.

        Args:
            sku: Product SKU
            requested_quantity: Quantity needed
            txn_id: Optional transaction ID

        Returns:
            dict: Availability status and details
        """
        product = self.get_product_info(sku, txn_id)
        if not product:
            return {
                "available": False,
                "reason": "Product not found",
                "requested": requested_quantity,
                "available_quantity": 0,
            }

        available_quantity = product["available"]
        is_available = available_quantity >= requested_quantity

        return {
            "available": is_available,
            "reason": "Sufficient stock" if is_available else "Insufficient stock",
            "requested": requested_quantity,
            "available_quantity": available_quantity,
            "total_quantity": product["quantity"],
            "reserved_quantity": product.get("reserved", 0),
        }

    def get_low_stock_products(
        self, threshold: int = 5, txn_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get products with stock levels below the specified threshold.

        Args:
            threshold: Minimum stock level threshold
            txn_id: Optional transaction ID

        Returns:
            dict: Products with low stock levels
        """
        low_stock = {}

        with self.db.lock:
            for sku, product in self.db.data.items():
                if isinstance(product, dict) and "quantity" in product:
                    available = product["quantity"] - product.get("reserved", 0)
                    if available <= threshold:
                        product_info = self.get_product_info(sku, txn_id)
                        if product_info:
                            low_stock[sku] = product_info

        return low_stock

    def generate_restock_report(self, txn_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a report of products that need restocking.

        Args:
            txn_id: Optional transaction ID

        Returns:
            dict: Restock report with recommendations
        """
        low_stock = self.get_low_stock_products(threshold=10, txn_id=txn_id)
        out_of_stock = self.get_low_stock_products(threshold=0, txn_id=txn_id)

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_products": len(self.db.data),
            "low_stock_count": len(low_stock),
            "out_of_stock_count": len(out_of_stock),
            "low_stock_products": low_stock,
            "out_of_stock_products": out_of_stock,
            "recommendations": self._generate_restock_recommendations(low_stock),
        }

    def _generate_restock_recommendations(
        self, low_stock_products: Dict[str, Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Generate restock quantity recommendations based on product data.

        Args:
            low_stock_products: Products with low stock

        Returns:
            dict: SKU to recommended restock quantity mapping
        """
        recommendations = {}

        for sku, product in low_stock_products.items():
            current_stock = product["available"]
            target_stock = 50  # Default target stock level

            if "MILK" in sku.upper():
                target_stock = 75
            elif "BREAD" in sku.upper():
                target_stock = 60

            recommended_quantity = max(0, target_stock - current_stock)
            if recommended_quantity > 0:
                recommendations[sku] = recommended_quantity

        return recommendations
