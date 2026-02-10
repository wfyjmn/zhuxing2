#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从对象存储恢复文件
尝试从Coze对象存储下载所有文件，恢复丢失的文件
"""

import os
from coze_coding_dev_sdk.s3 import S3SyncStorage

# 初始化对象存储客户端
storage = S3SyncStorage(
    endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
    access_key="",
    secret_key="",
    bucket_name=os.getenv("COZE_BUCKET_NAME"),
    region="cn-beijing",
)

def list_all_files():
    """列出所有对象存储文件"""
    print("=" * 80)
    print("正在列出对象存储中的所有文件...")
    print("=" * 80)

    all_keys = []
    continuation_token = None

    while True:
        result = storage.list_files(
            prefix="",
            max_keys=1000,
            continuation_token=continuation_token
        )

        all_keys.extend(result["keys"])
        print(f"\n当前批次获取到 {len(result['keys'])} 个文件")

        if not result["is_truncated"]:
            break

        continuation_token = result["next_continuation_token"]

    print(f"\n✅ 总计找到 {len(all_keys)} 个文件")
    print("=" * 80)

    # 打印所有文件
    print("\n文件列表：")
    for i, key in enumerate(all_keys, 1):
        print(f"{i:3d}. {key}")

    return all_keys

def restore_files(file_keys, target_dir="/workspace/projects/restored_files"):
    """恢复文件到本地"""
    print("\n" + "=" * 80)
    print("开始恢复文件...")
    print("=" * 80)

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    restored_count = 0
    failed_count = 0

    for key in file_keys:
        try:
            print(f"\n正在下载: {key}")

            # 读取文件内容
            content = storage.read_file(file_key=key)

            # 确定本地文件路径
            filename = os.path.basename(key)
            local_path = os.path.join(target_dir, filename)

            # 写入本地文件
            with open(local_path, 'wb') as f:
                f.write(content)

            print(f"✅ 已保存到: {local_path}")
            restored_count += 1

        except Exception as e:
            print(f"❌ 下载失败: {e}")
            failed_count += 1

    print("\n" + "=" * 80)
    print("恢复完成")
    print("=" * 80)
    print(f"成功恢复: {restored_count} 个文件")
    print(f"失败: {failed_count} 个文件")
    print(f"保存目录: {target_dir}")
    print("=" * 80)

def main():
    """主函数"""
    print("对象存储文件恢复工具")
    print("=" * 80)

    # 列出所有文件
    all_keys = list_all_files()

    if len(all_keys) == 0:
        print("\n⚠️  对象存储中没有找到任何文件")
        return

    # 恢复文件
    restore_files(all_keys)

if __name__ == '__main__':
    main()
