import requests

def upload_video(file_path):
    url = "http://172.16.186.168:7777/video2text/video_infer"
    with open(file_path, "rb") as file:
        files = {"file": ("boat.mp4", file, "video/mp4")}
        response = requests.post(url, files=files)
    
    return response.json()

if __name__ == "__main__":
    file_path = "/home/huilingb/xiaoheng/GenAIExamples/VideoRAGQnADistributed/vid2text/examples/boat.mp4"  # Replace with the actual path to your video file
    response = upload_video(file_path)
    print(response)
