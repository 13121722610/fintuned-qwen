# run_optimization.py
import sys
import argparse
from optimizer import MassMedicalDataOptimizer

def main():
    parser = argparse.ArgumentParser(description="批量优化医疗数据集")
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", default="data/data-optimized.json", help="输出文件路径")
    parser.add_argument("--api_key", required=True, help="API 密钥")
    parser.add_argument("--batch_size", type=int, default=100, help="每批大小")
    parser.add_argument("--max_workers", type=int, default=20, help="并发数")
    parser.add_argument("--request_delay", type=float, default=0.1, help="请求间隔秒数")

    args = parser.parse_args()

    optimizer = MassMedicalDataOptimizer(
        api_key=args.api_key,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        request_delay=args.request_delay
    )

    print("=" * 60)
    print("🚀 开始优化")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"批大小: {args.batch_size}")
    print(f"并发数: {args.max_workers}")
    print(f"请求间隔: {args.request_delay}秒")
    print("=" * 60)

    try:
        optimizer.process_all_data(args.input, args.output)
        print("\n✅ 优化完成")
    except Exception as e:
        print(f"❌ 出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()