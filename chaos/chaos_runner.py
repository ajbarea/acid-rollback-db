import time
import random

from src.database import AcidRollbackDB
from .exceptions.chaos_exception import ChaosException
from .chaos_config import ChaosConfig
from .chaos_proxy import ChaosProxy


#initalize the chaos configuration
chaos = ChaosConfig(enabled = True, failure_rate=0.3, delay_chance=0.4, max_delay=1.5)

#initialize the database
real_db = AcidRollbackDB()

#initialize the chaos proxy with the database and chaos configuration
db = ChaosProxy(real_db, chaos)

#run the chaos runner in a loop
start_time = time.time()
duration = 10 # Run for 10 seconds

success_insert_keys = []
keys = [f"item_{i}" for i in range(10)]


while time.time() - start_time < duration:
    try:
        #begin the transaction
        transaction = db.begin_transaction()
        print(f"[CHAOS TEST] Started txn {transaction}")

        for _ in range(1,3):
            #randomly choose to put or delete a key
            action = random.choice(['put', 'delete'])
            key = random.choice(keys)

            if action == 'put':
                value  = {"value": random.randint(1, 100)}
                db.put(transaction, key, value)
                success_insert_keys.append(key)
                print(f"[CHAOS TEST] PUT {key} = {value}")
            elif action == 'delete':
                #use random to decided whether to pick a key from the success_insert_keys keys pool or use a new key
                use_success_insert_keys = random.random() < 0.5 and success_insert_keys
                if use_success_insert_keys:
                    key = random.choice(success_insert_keys)
                db.delete(transaction, key)
                if key in success_insert_keys:
                    success_insert_keys.remove(key)
                print(f"[CHAOS TEST] DELETE {key}")
        
        #randomly decide to commit or rollback the transaction
        if random.random() < 0.7:
            db.commit(transaction)
            print(f"[CHAOS TEST] Committed txn {transaction}")
        else:
            db.rollback(transaction)
            print(f"[CHAOS TEST] Rolled back txn {transaction}")
    except ChaosException as e:
        print(f"[CHAOS TEST] ChaosException occurred: {e}")
    except Exception as e:
        print(f"[CHAOS TEST] Exception occurred: {e}")
    

    time.sleep(1)  # Sleep to simulate time between operations


print("[CHAOS TEST] Final database state:")
db.print_status()
print("[CHAOS TEST] Chaos runner completed.")


print("\n[CHAOS TEST] Chaos metrics:")
db.print_chaos_metrics()



