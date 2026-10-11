import argparse
from .pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description="NeuroCurate image-level RDM analysis")
    parser.add_argument("config", help="YAML experiment configuration")
    args = parser.parse_args()
    output = run_experiment(args.config)
    print(f"Completed analysis: {output}")


if __name__ == "__main__":
    main()
