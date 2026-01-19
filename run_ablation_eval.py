from scripts.ablation_eval import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模态消融评估脚本")
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径 (.pth)')
    parser.add_argument('--image_dir', type=str, default='', help='测试集图像文件夹路径')
    parser.add_argument('--json_path', type=str, default='', help='测试集JSON路径')
    parser.add_argument('--config', type=str, default='config.yml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='results/ablation', help='结果输出目录')
    args = parser.parse_args()
    main(args)
