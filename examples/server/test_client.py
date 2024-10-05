import requests, json, base64
from io import BytesIO
from PIL import Image, PngImagePlugin, ImageShow
from threading import Thread

def save_img(img: Image,path: str) -> None:
    info = PngImagePlugin.PngInfo()
    for key, value in img.info.items():
        info.add_text(key, value)
    img.save(path,pnginfo=info)

def show_img(img: Image, title = None) -> None:
    info = PngImagePlugin.PngInfo()
    for key, value in img.info.items():
        info.add_text(key, value)
    tmp = img._dump(title, format=img.format, pnginfo=info)
    print(f"Image path: {tmp}\n")
    for viewer in ImageShow._viewers:
        if viewer.show_file(tmp,title=title):
            return

_protocol = "http"
_server = "localhost"
_port = 8080
_endpoint = "txt2img"

def update_url(protocol= None, server=None,port=None,endpoint=None):
    global _protocol, _server, _port, _endpoint
    if protocol:
        _protocol = protocol
    if server:
        _server = server
    if port:
        _port = port
    if endpoint:
        _endpoint = endpoint
    return f"{_protocol}://{_server}:{_port}/{_endpoint}"

url = update_url(port=8084)

def sendRequest(payload):
    global url
    return requests.post(url,payload ).text

def getImages(response):
    return [Image.open(BytesIO(base64.b64decode(img["data"]) )) for img in json.loads(response)]

def showImages(imgs):
    for img in imgs:
        t =Thread(target=show_img,args=(img,))
        t.setDaemon(True)
        t.start()
    

def print_usage():
    print("""Example usage (images will be displayed and saved to a temporary file):
showImages(getImages(sendRequest(json.dumps({'seed': -1, 'batch_count':4, 'sample_steps':24, 'width': 512, 'height':768, 'negative_prompt': "Bad quality", 'prompt': "A beautiful image"}))))""")
    
if __name__ == "__main__":
    print_usage()
