import argparse
import requests
import numpy as np
import json
import grpc
from hsmlib import hsm_srv_pb2
from hsmlib import hsm_srv_pb2_grpc
import os
import glob
import shutil
import sys
def hsm_print(str):
    print("****hsm_debug: "+str)

def is_dir_empty(dir_path):
    # 检查目录是否存在
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} 不是一个有效的目录路径")
    
    # 列出目录中的所有文件和子目录
    if not os.listdir(dir_path):
        return True
    else:
        return False
    
class HSMClient():
    def __init__(self, host, port):
        self.table2handle = {}
        ipaddr = host+":"+str(port)
        self.channel = grpc.insecure_channel(ipaddr)
        self.stub = hsm_srv_pb2_grpc.HSMServiceStub(self.channel)

    def add_table(self, table_name,data_type,dimension,persist_dic):
        # table_path = persist_dic+table_name
        # index_pattern =table_path+'/*disk.index'
        # pq_pattern = table_path+'/*pq_compressed.bin'
        # pq_pattern2 = table_path+'/*pq_pivots.bin'
        # matching_files1 = glob.glob(index_pattern)
        # matching_files2 = glob.glob(pq_pattern)
        # matching_files3 = glob.glob(pq_pattern2)
        # if os.path.exists(table_path) and matching_files1 != [] and matching_files2 != [] and matching_files3 != []:
        #     self.load_table_loop(table_path)
        # elif not os.path.exists(table_path):
        #     os.makedirs(table_path)
        #     hsm_print("make dic")
        #     self.create_table_loop(table_path,data_type,dimension,table_name)
        # else:
        #     shutil.rmtree(table_path)
        #     os.makedirs(table_path)
        #     self.create_table_loop(table_path,data_type,dimension,table_name)
        table_path = persist_dic+table_name
        hsm_print(table_path)
        # index_pattern =table_path+'/*disk.index'
        # pq_pattern = table_path+'/*pq_compressed.bin'
        # pq_pattern2 = table_path+'/*pq_pivots.bin'
        # matching_files1 = glob.glob(index_pattern)
        # matching_files2 = glob.glob(pq_pattern)
        # matching_files3 = glob.glob(pq_pattern2)
        if os.path.exists(table_path) and not is_dir_empty(table_path):
            hsm_print("---------------= Try to load table =----------------")
            self.load_table_loop(table_path)
            hsm_print("---------------= Successfully load table =----------------")
        else:
            hsm_print("---------------= creat table =----------------")
            os.makedirs(table_path, exist_ok=True)
            if os.path.exists(table_path):
                hsm_print("create ok1")
            else:
                hsm_print("create no")

            self.create_table_loop(table_path,data_type,dimension,table_name)
            hsm_print("---------------= creat table successfully=----------------")
        # else:
        #     shutil.rmtree(table_path)
        #     os.makedirs(table_path)
        #     self.create_table_loop(table_path,data_type,dimension,table_name)


    def load_table_loop(self,table_path):
        request = hsm_srv_pb2.LoadRequest(db_dir=table_path)
        response = self.stub.Load(request)

        if response.db_handle > 0:
            hsm_print(f"Load table successful. Index handle: {response.db_handle}\n")
            last_part = table_path.split('/')[-1]
            hsm_print(f"table name: {last_part}\n")
            self.table2handle[last_part] = response.db_handle
        else:
            hsm_print("Load table failed.\n")

    def create_table_loop(self,db_dir,data_type,dimension,table_name):
        if data_type =="float":
            dt = hsm_srv_pb2.DT.DT_FLOAT
        elif data_type == "int8":
            dt = hsm_srv_pb2.DT.DT_INT8
        elif data_type == "uint8":
            dt = hsm_srv_pb2.DT.DT_UINT8
        else:
            dt = None
            hsm_print("unsupport type")
        request = hsm_srv_pb2.CreateRequest(db_dir=db_dir, data_type=dt, dimension=dimension)
        response = self.stub.Create(request)
        
        if response.db_handle > 0:
            self.table2handle[table_name] = response.db_handle
            hsm_print(f"Create table successful. Index handle: {response.db_handle}")
        else:
            hsm_print("Create table failed.")

    def close_table_loop(self,table_name):
        if table_name in self.table2handle.keys():
            handle = self.table2handle[table_name]
            request = hsm_srv_pb2.CloseRequest(index_handle=handle)
            response = self.stub.Close(request)

            if response.index_handle > 0:
                hsm_print(f"Close table successful. Index handle: {response.index_handle}\n")
            else:
                hsm_print("Close table failed.\n")
        else:
            hsm_print("Table name is not exist.\n")

    def convert_to_np_array(self, numpy_array):
        numpy_array = np.array(numpy_array)
        np_array = hsm_srv_pb2.NPArray()
        if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
            np_array.float_elems.extend(numpy_array.flatten())
        elif numpy_array.dtype == np.uint8 or numpy_array.dtype == np.int8:
            np_array.bytes_elem = numpy_array.tobytes() # test
        else:
            raise ValueError("Unsupported numpy array data type\n")
        if len(numpy_array.shape) == 1:
            np_array.npts = 1
            np_array.dims = numpy_array.shape[0]
        else:          
            np_array.npts = numpy_array.shape[0]
            np_array.dims = numpy_array.shape[1]

        return np_array

    def add_text_loop(self,table_name,vector_ids,np_array):
        if table_name in self.table2handle.keys():
            handle = self.table2handle[table_name]
            converted_np_array = self.convert_to_np_array(np_array)
            # hsm_print(vector_ids)
            request = hsm_srv_pb2.AddRequest(db_handle=handle, vector_ids=vector_ids, vectors=converted_np_array)
            response = self.stub.Add(request)
            if response.ok_added == len(np_array):
                hsm_print(f"Successfully add: {response.ok_added}")
            else:
                hsm_print("Add text failed.")
        else:
            hsm_print("add text, Table name is not exist.")

    def build_index(self, table_name, build_param):
        db_handle = self.table2handle[table_name]
        if build_param is not None:
            if build_param["dist_fn"] == "l2":
                dist = hsm_srv_pb2.DF.DF_L2
            elif build_param["dist_fn"] == "IP":
                dist = hsm_srv_pb2.DF.DF_IP
            elif build_param["dist_fn"] == "cosine":
                dist = hsm_srv_pb2.DF.DF_Cosine
            else:
                dist = hsm_srv_pb2.DF.DF_Cosine
            hsm_build_param = hsm_srv_pb2.BuildRequest.BuildParam(
                dist_fn=dist,
                R=int(build_param["R"]),
                L=int(build_param["L"]),
                B=int(build_param["B"]),
                T=int(build_param["T"]),
                QD=int(build_param["QD"])
            )
        else:
            hsm_build_param = None
        request = hsm_srv_pb2.BuildRequest(db_handle=db_handle, param=hsm_build_param)
        response = self.stub.Build(request)
        if response.result:
            hsm_print(f"Build index successful.")
        else:
            hsm_print("Build index failed.")

    def query_loop(self,table_name, query_vs, Ls, k_value, debug):
        res_idx = []
        debug_info_str = ""
        if table_name in self.table2handle.keys():
            handle = self.table2handle[table_name]
            query_vs_np = self.convert_to_np_array(query_vs)
            query_request = hsm_srv_pb2.QueryRequest(db_handle=handle, vectors=query_vs_np, k=k_value, Ls=Ls, debug=debug)
            response = self.stub.Query(query_request)
            for i, knn in enumerate(response.nns):
                res_idx.append(knn.vector_ids)
                if debug:
                    for j, debug_info in enumerate(knn.debug_info):
                        debug_info_str += f" Debug Info {j}:\n"
                        debug_info_str += f" Index: {debug_info.index}\n"
                        debug_info_str += f" Distance: {debug_info.dist}\n"
                        debug_info_str += f" Latency (us): {debug_info.latency_us}\n"

            # self.print_query_response(response)

        else:
            hsm_print("query, Table name is not exist.")
        return res_idx,debug_info_str

    def table_status(self,table_name):
        if table_name == None:
            request = hsm_srv_pb2.StatusRequest()
        else:
            if table_name in self.table2handle.keys():
                db_handles = self.table2handle[table_name]
                request = hsm_srv_pb2.StatusRequest(db_handle=db_handles)
                # hsm_print(db_handle)
            else:
                hsm_print("table is not exists.")
                return None
        response = self.stub.Status(request)
        self.print_status_response(response)
        return response

    def print_query_response(self, response):
        hsm_print("QueryResponse:")
        for i, knn in enumerate(response.nns):
            hsm_print(f"kNN {i}:")
            hsm_print(f"Vector IDs: {knn.vector_ids}")
            for j, debug_info in enumerate(knn.debug_info):
                hsm_print(f" Debug Info {j}:")
                hsm_print(f" Index: {debug_info.index}")
                hsm_print(f" Distance: {debug_info.dist}")
                hsm_print(f" Latency (us): {debug_info.latency_us}")

    def print_status_response(self,response):
        hsm_print("Response:")
        hsm_print(response)
        # for db_status in response.db_status:
        #     hsm_print(f" DB Directory: {db_status.db_dir}")
        #     hsm_print(f" DB Handle: {db_status.db_handle}")
        #     hsm_print(f" Status: {hsm_srv_pb2.StatusResponse.DB_STATUS.Name(db_status.status)}")
        #     hsm_print(f" Cache Size: {db_status.cache_size}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Huashan Vector DB client")
    parser.add_argument("--config_file",help="Path of config file")
    args = parser.parse_args()


    try:
        with open(args.config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        
        if isinstance(config, (dict, list)):
            hsm_print("JSON config loaded.")
            # hsm_print(config)
        else:
            hsm_print("JSON config with wrong format...")
        
    except json.JSONDecodeError as e:
        hsm_print(f"JSON failed load: {e}")
    except FileNotFoundError:
        hsm_print(f"JSON file missed: {args.config_file}")
    except Exception as e:
        hsm_print(f"Error: {e}")

    hsm = HSMClient("127.0.0.1:10027")
    # hsm.add_table
    # elif config["operation"] == "add_table":
    #     required_fields = ["table_name"]
    #     missing_fields = [field for field in required_fields if field not in config]
    #     if missing_fields:
    #         hsm_print(f"miss config variable: {', '.join(missing_fields)}")
        
    #     table_name = config["table_name"]
    #     table_path = "/home/cheny/Project/langchain/UTL_Diskann/DiskANN/build/data/"+table_name
    #     if os.path.exists(table_path):
    #         hsm.load_table_loop(table_path)
    #     else:
    #         hsm.create_table_loop(table_path,table_name)

    # elif config["operation"] == "delete":
    #     required_fields = ["table_name"]
    #     missing_fields = [field for field in required_fields if field not in config]
    #     if missing_fields:
    #         hsm_print(f"miss config variable: {', '.join(missing_fields)}")
        
    #     table_name = config["table_name"]
    #     hsm.close_table_loop(table_name)
    # elif config["operation"] == "add_texts" or config["operation"] == "add_images":
    #     pass
    # elif config["operation"] == "build":
    #     pass
    # else:
    #     hsm_print("Unsupport operation.")
