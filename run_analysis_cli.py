from scripts.run_analysis import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型分析工具：Grad-CAM 和 特征秩分析")
    parser.add_argument('--image_dir', type=str, required=True, help='测试集图像文件夹路径')
    parser.add_argument('--json_path', type=str, required=True, help='测试集JSON路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='分析结果输出目录')
    parser.add_argument('--config', type=str, default='config.yml', help='配置文件路径')
    parser.add_argument(
        '--ablation_mode',
        type=str,
        default='',
        help='Ablation: none | image_only | text_off (default: none)',
    )

    args = parser.parse_args()
    main(args)
