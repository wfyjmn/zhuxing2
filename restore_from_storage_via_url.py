#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从对象存储恢复文件（使用签名URL）
尝试通过签名URL下载文件
"""

import os
import requests
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

        if not result["is_truncated"]:
            break

        continuation_token = result["next_continuation_token"]

    print(f"✅ 总计找到 {len(all_keys)} 个文件\n")

    # 过滤掉目录
    file_keys = [key for key in all_keys if not key.endswith('/')]

    print("文件列表：")
    for i, key in enumerate(file_keys, 1):
        print(f"{i:3d}. {key}")

    return file_keys

def restore_files_via_url(file_keys, target_dir="/workspace/projects/restored_files"):
    """通过签名URL恢复文件"""
    print("\n" + "=" * 80)
    print("开始恢复文件（通过签名URL）...")
    print("=" * 80)

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    restored_count = 0
    failed_count = 0

    for key in file_keys:
        try:
            print(f"\n正在处理: {key}")

            # 生成签名URL（有效期1小时）
            signed_url = storage.generate_presigned_url(key=key, expire_time=3600)
            print(f"  签名URL: {signed_url}")

            # 下载文件
            response = requests.get(signed_url, timeout=60)

            if response.status_code == 200:
                # 确定本地文件路径
                filename = os.path.basename(key)
                local_path = os.path.join(target_dir, filename)

                # 写入本地文件
                with open(local_path, 'wb') as f:
                    f.write(response.content)

                file_size = len(response.content)
                file_size_mb = file_size / (1024 * 1024)
                print(f"✅ 已保存到: {local_path}")
                print(f"   文件大小: {file_size_mb:.2f} MB")
                restored_count += 1
            else:
                print(f"❌ 下载失败: HTTP {response.status_code}")
                print(f"   响应: {response.text[:200]}")
                failed_count += 1

        except Exception as e:
            print(f"❌ 处理失败: {e}")
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
    print("对象存储文件恢复工具（使用签名URL）")
    print("=" * 80)

    # 列出所有文件
    all_keys = list_all_files()

    if len(all_keys) == 0:
        print("\n⚠️  对象存储中没有找到任何文件")
        return

    # 恢复文件
    restore_files_via_url(all_keys)

if __name__ == '__main__':
    main()
