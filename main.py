"""Main entry point for the Regional Supermarket Inventory Tracking System."""

from src.database import AcidRollbackDB


def main():
    """Demonstrate availability tactics in a regional supermarket inventory system."""
    # Create store inventory database
    store_db = AcidRollbackDB(checkpoint_interval=5, max_checkpoints=20)

    print("=== REGIONAL SUPERMARKET INVENTORY SYSTEM ===")
    print("Store: WESTSIDE-001 | Distribution Center Integration\n")

    # Test 1: Initial inventory setup
    print("1. Setting up initial store inventory...")
    setup_txn = store_db.begin_transaction()

    # Dairy products
    store_db.put(
        setup_txn,
        "MILK-2PCT-1GAL",
        {
            "sku": "MILK-2PCT-1GAL",
            "product_name": "2% Milk - 1 Gallon",
            "category": "Dairy",
            "quantity": 48,
            "reserved": 0,
            "price": 3.99,
            "min_stock": 12,
            "store_id": "WESTSIDE-001",
            "last_updated": "2025-01-22T08:00:00Z",
        },
    )

    store_db.put(
        setup_txn,
        "BREAD-WHITE-LOAF",
        {
            "sku": "BREAD-WHITE-LOAF",
            "product_name": "White Bread Loaf",
            "category": "Bakery",
            "quantity": 36,
            "reserved": 0,
            "price": 2.49,
            "min_stock": 8,
            "store_id": "WESTSIDE-001",
            "last_updated": "2025-01-22T08:00:00Z",
        },
    )

    store_db.put(
        setup_txn,
        "EGGS-LARGE-DOZEN",
        {
            "sku": "EGGS-LARGE-DOZEN",
            "product_name": "Large Eggs - Dozen",
            "category": "Dairy",
            "quantity": 24,
            "reserved": 0,
            "price": 4.29,
            "min_stock": 6,
            "store_id": "WESTSIDE-001",
            "last_updated": "2025-01-22T08:00:00Z",
        },
    )

    store_db.commit_transaction(setup_txn)
    print("✓ Initial inventory loaded successfully")
    store_db.print_status()

    # Test 2: Customer purchase transaction with rollback scenario
    print("2. Processing customer purchase transaction...")
    purchase_txn = store_db.begin_transaction()

    try:
        # Customer wants: 2 gallons milk, 1 bread, 2 dozen eggs
        print("   Customer purchase: 2x Milk, 1x Bread, 2x Eggs")

        # Check stock availability first
        milk_check = store_db.inventory.check_stock_availability("MILK-2PCT-1GAL", 2)
        bread_check = store_db.inventory.check_stock_availability("BREAD-WHITE-LOAF", 1)
        eggs_check = store_db.inventory.check_stock_availability("EGGS-LARGE-DOZEN", 2)

        if not all(
            [milk_check["available"], bread_check["available"], eggs_check["available"]]
        ):
            raise ValueError("Insufficient stock for customer purchase")

        # Process sale
        store_db.inventory.update_stock_level(
            purchase_txn, "MILK-2PCT-1GAL", -2, "sale"
        )
        store_db.inventory.update_stock_level(
            purchase_txn, "BREAD-WHITE-LOAF", -1, "sale"
        )
        store_db.inventory.update_stock_level(
            purchase_txn, "EGGS-LARGE-DOZEN", -2, "sale"
        )

        store_db.commit_transaction(purchase_txn)
        print("✓ Customer purchase completed successfully")

    except Exception as e:
        print(f"✗ Purchase failed: {e}")
        store_db.rollback_transaction(purchase_txn)
        print("✓ Transaction rolled back - inventory unchanged")

    store_db.print_status()

    # Test 3: Online order reservation with potential rollback
    print("3. Processing online order reservation...")
    reservation_txn = store_db.begin_transaction()

    try:
        print("   Online order: Reserve 3x Milk, 2x Bread for pickup")

        # Reserve inventory for online pickup
        store_db.inventory.reserve_inventory(reservation_txn, "MILK-2PCT-1GAL", 3)
        store_db.inventory.reserve_inventory(reservation_txn, "BREAD-WHITE-LOAF", 2)

        store_db.commit_transaction(reservation_txn)
        print("✓ Online order reserved successfully")

    except Exception as e:
        print(f"✗ Reservation failed: {e}")
        store_db.rollback_transaction(reservation_txn)
        print("✓ Reservation rolled back")

    print("Before rollback:")
    store_db.print_status()

    # Test 4: Simulate POS system crash during busy period
    print("4. Simulating POS system crash during peak hours...")

    # Start multiple transactions (busy checkout period)
    busy_txns = []
    for i in range(3):
        txn = store_db.begin_transaction()
        store_db.inventory.update_stock_level(
            txn, "MILK-2PCT-1GAL", -1, f"sale_batch_{i}"
        )
        busy_txns.append(txn)
        print(f"   Started checkout transaction {i+1}")

    print("   SYSTEM FAULT DETECTED!")
    print("   Activating fault recovery protocol...")

    # Simulate system crash - rollback to last stable checkpoint
    store_db.simulate_fault()

    print("✓ System recovered to last stable checkpoint")
    print("✓ All incomplete transactions aborted")
    store_db.print_status()

    # Test 5: Inventory audit and adjustment
    print("5. Performing daily inventory audit...")
    audit_txn = store_db.begin_transaction()

    try:
        # Physical count shows discrepancy
        print("   Physical count: Milk inventory shows 41 units (expected 43)")
        store_db.inventory.adjust_inventory(
            audit_txn, "MILK-2PCT-1GAL", 41, "physical_audit"
        )

        store_db.commit_transaction(audit_txn)
        print("✓ Inventory audit completed - discrepancy recorded")

    except Exception as e:
        print(f"✗ Audit failed: {e}")
        store_db.rollback_transaction(audit_txn)

    # Test 6: Generate restock report for distribution center
    print("6. Generating restock report for distribution center...")

    restock_report = store_db.inventory.generate_restock_report()
    print(f"   Products needing restock: {restock_report['low_stock_count']}")
    print(f"   Out of stock products: {restock_report['out_of_stock_count']}")

    if restock_report["recommendations"]:
        print("   Restock recommendations:")
        for sku, qty in restock_report["recommendations"].items():
            product = store_db.inventory.get_product_info(sku)
            print(f"     {product['product_name']}: Order {qty} units")

    print("\n=== FINAL SYSTEM STATUS ===")
    store_db.print_status()

    print("\n=== AVAILABILITY TACTICS DEMONSTRATION COMPLETED ===")
    print("✓ Fault Prevention: ACID transactions ensured data consistency")
    print("✓ Fault Recovery: Rollback mechanisms maintained system availability")
    print("✓ Store inventory remains accurate and available for operations")


if __name__ == "__main__":
    main()
