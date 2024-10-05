import requests, json, base64
from io import BytesIO
from PIL import Image, PngImagePlugin, ImageShow
import os
if os.name == 'nt':
    from threading import Thread


from typing import List


def save_img(img: Image, path: str) -> None:
    """
    Save the image to the specified path with metadata.

    Args:
    img (Image): The image to be saved.
    path (str): The path where the image will be saved.

    Returns:
    None
    """
    info = PngImagePlugin.PngInfo()
    for key, value in img.info.items():
        info.add_text(key, value)
    img.save(path, pnginfo=info)

def show_img(img: Image, title = None) -> None:
    """
    Display the image (with metadata) in a new window and print the path of the temporary file.

    Args:
    img (Image): The image to be displayed.
    title (str, optional): The title of the image window. Defaults to None.

    Returns:
    None
    """
    info = PngImagePlugin.PngInfo()
    for key, value in img.info.items():
        info.add_text(key, value)
    tmp = img._dump(None, format=img.format, pnginfo=info)
    print(f"Image path: {tmp}\n")
    for viewer in ImageShow._viewers:
        if viewer.show_file(tmp,title=title):
            return

_protocol = "http"
_server = "localhost"
_port = 8080
_endpoint = "txt2img"
url=""

def update_url(protocol=None, server=None, port=None, endpoint=None) -> str:
    """
    Update the global URL variable with the provided protocol, server, port, and endpoint.

    This function takes optional arguments for protocol, server, port, and endpoint.
    If any of these arguments are provided, the corresponding global variable is updated with the new value.
    The function then constructs the URL using the updated global variables and returns it.

    Args:
    protocol (str, optional): The protocol to be used in the URL. Defaults to None.
    server (str, optional): The server address to be used in the URL. Defaults to None.
    port (int, optional): The port number to be used in the URL. Defaults to None.
    endpoint (str, optional): The endpoint to be used in the URL. Defaults to None.

    Returns:
    str: The updated URL.
    """
    global _protocol, _server, _port, _endpoint, url
    if protocol:
        _protocol = protocol
    if server:
        _server = server
    if port:
        _port = port
    if endpoint:
        _endpoint = endpoint
    url = f"{_protocol}://{_server}:{_port}/{_endpoint}"
    return url

# set default url value
update_url()

def sendRequest(payload: str) -> str:
    """
    Send a POST request to the API endpoint with the provided payload.

    This function takes a payload as input and sends a POST request to the API endpoint specified by the global URL variable.
    The function then returns the text content of the response.

    Args:
    payload (str): The payload to be sent in the POST request.

    Returns:
    str: The text content of the response from the POST request.
    """
    global url
    return requests.post(url, payload).text

def getImages(response: str) -> List[Image.Image]:
    """
    Convert base64 encoded image data from the API response into a list of Image objects.

    This function takes the text response from the API as input and parses it as JSON.
    It then iterates over each image data in the JSON response, decodes the base64 encoded image data,
    and uses the BytesIO class to convert it into a PIL Image object.
    The function returns a list of these Image objects.

    Args:
    response (str): The text response from the API containing base64 encoded image data.

    Returns:
    List[Image.Image]: A list of PIL Image objects decoded from the base64 encoded image data in the API response.
    """
    return [Image.open(BytesIO(base64.b64decode(img["data"]))) for img in json.loads(response)]

def showImages(imgs: List[Image.Image]) -> None:
    """
    Display a list of images in separate threads.

    This function takes a list of PIL Image objects as input and creates a new thread for each image.
    Each thread calls the show_img function to display the image in a new window and print the path of the temporary file.
    The function does not return any value.

    Args:
    imgs (List[Image.Image]): A list of PIL Image objects to be displayed.

    Returns:
    None
    """
    for (i,img) in enumerate(imgs):
        if os.name == 'nt':
            t = Thread(target=show_img, args=(img, f"IMG {i}"))
            t.daemon = True
            t.start()
        else:
            show_img(img, f"IMG {i}")

def saveImages(imgs: List[Image.Image], path: str) -> None:
    """
    Save a list of images to the specified path with metadata.

    This function takes a list of PIL Image objects and a path as input.
    For each image, it calls the save_img function to save the image to a file
    with the name "{path}{i}.png", where i is the index of the image in the list.
    The function does not return any value.

    Args:
    imgs (List[Image.Image]): A list of PIL Image objects to be saved.
    path (str): The path where the images will be saved.

    Returns:
    None
    """
    if path.endswith(".png"):
        path = path[:-4]
    for (i, img) in enumerate(imgs):
        save_img(img, f"{path}{i}.png")
    

def _print_usage():
    print("""Example usage (images will be displayed and saved to a temporary file):
update_url(server="127.0.0.1", port=8080)
showImages(getImages(sendRequest(json.dumps({'seed': -1, 'batch_count':4, 'sample_steps':24, 'width': 512, 'height':768, 'negative_prompt': "Bad quality", 'prompt': "A beautiful image"}))))""")
    
if __name__ == "__main__":
    _print_usage()
