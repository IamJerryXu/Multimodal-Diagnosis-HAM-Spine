from scripts.predict import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="课程大作业模型预测脚本",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--image_dir', type=str, required=True, help='【必需】测试集图像文件夹的路径。')
    parser.add_argument('--json_path', type=str, required=True, help='【必需】测试集JSON元数据文件的路径。')
    parser.add_argument('--model_path', type=str, required=True, help='【必需】您训练好的模型权重文件（.pth文件）的路径。')
    parser.add_argument('--output_path', type=str, required=True, help='【必需】预测结果的输出路径（例如：./submission.csv）。')
    parser.add_argument('--config', type=str, default='config.yml', help='【可选】模型配置文件路径，默认为 "config.yml"。')

    args = parser.parse_args()
    main(args)
