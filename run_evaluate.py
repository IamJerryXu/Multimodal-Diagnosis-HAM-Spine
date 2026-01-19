from scripts.evaluate import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="在隐藏测试集上评估学生模型")
    parser.add_argument('--model_path', type=str, required=True, help='学生提交的模型权重文件路径 (.pth)')
    parser.add_argument('--test_image_dir', type=str, required=True, help='隐藏测试集的图像文件夹路径')
    parser.add_argument('--test_json_path', type=str, required=True, help='隐藏测试集的JSON元数据文件路径')
    parser.add_argument('--config', type=str, default='config.yml', help='用于模型架构的配置文件路径')

    args = parser.parse_args()
    main(args)
