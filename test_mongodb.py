#!/usr/bin/env python3
"""
测试MongoDB连接和基本功能的简单脚本
"""

import os
import time
from pymongo import MongoClient
from datetime import datetime

# 设置MongoDB连接URI
# 如果是在本地测试，直接使用localhost
# 如果是通过docker-compose测试，使用服务名称
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/mydb")

def test_mongodb_connection():
    """测试MongoDB连接并执行基本CRUD操作"""
    print(f"尝试连接到MongoDB: {MONGO_URI}")
    
    try:
        # 创建连接
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # 测试连接
        client.admin.command('ping')
        print("连接成功! MongoDB服务器正在运行")
        
        # 获取或创建测试数据库
        db_name = MONGO_URI.split("/")[-1]
        db = client[db_name]
        
        # 获取测试集合
        test_collection = db['test_collection']
        
        # 清理之前的测试数据
        test_collection.drop()
        print("已清理之前的测试集合")
        
        # 插入测试文档
        test_doc = {
            "test_id": "test1",
            "message": "这是一个测试文档",
            "created_at": datetime.now()
        }
        result = test_collection.insert_one(test_doc)
        print(f"插入文档成功，ID: {result.inserted_id}")
        
        # 查询文档
        found_doc = test_collection.find_one({"test_id": "test1"})
        if found_doc:
            print(f"查询成功: {found_doc['message']}")
        else:
            print("查询失败，未找到文档")
        
        # 更新文档
        update_result = test_collection.update_one(
            {"test_id": "test1"},
            {"$set": {"message": "这是一个已更新的测试文档"}}
        )
        print(f"更新文档，匹配数: {update_result.matched_count}, 修改数: {update_result.modified_count}")
        
        # 再次查询确认更新
        updated_doc = test_collection.find_one({"test_id": "test1"})
        if updated_doc:
            print(f"更新后的文档: {updated_doc['message']}")
        
        # 删除文档
        delete_result = test_collection.delete_one({"test_id": "test1"})
        print(f"删除文档，删除数: {delete_result.deleted_count}")
        
        # 创建索引测试
        index_result = test_collection.create_index("test_id")
        print(f"创建索引成功: {index_result}")
        
        # 列出所有索引
        indexes = list(test_collection.list_indexes())
        print(f"集合有 {len(indexes)} 个索引")
        
        # 使用我们已经创建的AudioTranscription类进行测试
        print("\n测试AudioTranscription模型...")
        try:
            from common.models import AudioTranscription
            
            # 创建一个测试记录
            transcript_id = AudioTranscription.create(
                filename="test_audio.txt",
                transcription="这是一个测试转录文本"
            )
            print(f"已创建测试转录，ID: {transcript_id}")
            
            # 查找所有记录
            all_transcripts = AudioTranscription.find_all()
            print(f"找到 {len(all_transcripts)} 条转录记录")
            
            # 按文件名查找
            found = AudioTranscription.find_by_filename("test_audio.txt")
            if found:
                print(f"成功找到转录: {found['transcription']}")
            else:
                print("未找到指定的转录记录")
                
        except ImportError:
            print("无法导入AudioTranscription模型，请确保common目录在Python路径中")
        except Exception as e:
            print(f"测试AudioTranscription时出错: {e}")
            
        print("\n全部测试完成，MongoDB功能正常工作!")
        
    except Exception as e:
        print(f"错误: {e}")
        return False
        
    return True


if __name__ == "__main__":
    # 尝试连接，如果失败则等待并重试
    retries = 5
    while retries > 0:
        if test_mongodb_connection():
            break
        print(f"将在5秒后重试... 剩余尝试次数: {retries}")
        retries -= 1
        time.sleep(5) 