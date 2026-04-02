import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import flwr as fl
import os
import sys

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

try:
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(
            num_rounds=10,
            round_timeout=300
        ),
        strategy=strategy,
    )
    print("\n✅ Federated Training Complete! All 10 rounds finished.")

except KeyboardInterrupt:
    print("\n✅ Server stopped by user")

finally:
    print("✅ Server shut down successfully")
    os._exit(0)  # Force clean exit - kills grpc threads immediately