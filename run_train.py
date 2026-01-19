from scripts.train import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练入口")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="配置文件路径（默认: config.yml）",
    )
    args = parser.parse_args()
    main(args.config)
