# curl -X 'POST' \
#   'http://172.16.186.168:7000/agent/prompt' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "prompt": "string"
# }'

# import requests

# def post():
#     url = "http://172.16.186.168:7000/agent/prompt"
#     headers = {'Content-Type': 'application/json'}
#     body = {"prompt": "string"}
#     response = requests.post(url, json=body, headers=headers)
#     if response.status_code != 200:
#         print(f"Request failed with status code {response.status_code}")
#     print(response.status_code)
#     return response.json()

# if __name__ == "__main__":
#     response = post()
#     print(response)


import requests
ip = "172.16.186.168"
prompt_tmd = {"query":"A man with shirt"} #a man holding a red basket
print(prompt_tmd)
print(type(prompt_tmd))
# 发送 POST 请求，并添加 headers
headers = {
    'Content-Type': 'application/json',
}
response = requests.post("http://"+ip+":7000/agent/prompt", json=prompt_tmd,headers=headers)
    # 检查请求是否成功，并提取 JSON 数据
result = response.json()
print(result)
if response.status_code == 200:
    result = response.json()  # 提取 JSON 数据
    prompt = result.get('prompt')  # 提取 prompt 字段
    video_name = result.get('video')  # 提取 video_url 字段
    print(result)
else:
    # 处理错误情况，例如返回错误消息或默认值
    prompt = "Error fetching prompt"
    video_name = "Error fetching video_url"
    print("result")