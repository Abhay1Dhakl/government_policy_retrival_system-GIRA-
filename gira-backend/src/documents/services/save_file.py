import base64
import gdown
import os
import re
import requests
from urllib.parse import urlparse, parse_qs

from django.core.files.base import ContentFile

from src.constants.database_constants import FileSource


def fetch_and_save_file(file_url, file_source):

    if file_source == FileSource.GOOGLE_DRIVE:
        file_obj = save_google_drive(file_url)
        return file_obj

    # if file_source == FileSource.ONEDRIVE:
    #     return save_onedrive_file(file_url)

    access_token = "your_access_token"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(file_url, headers=headers)

    if response.status_code == 200:
        file_name = file_url.split("/")[-1]
        file_obj = ContentFile(response.content, name=file_name)
        return file_obj

    else:
        raise Exception("Failed to fetch file from external storage")


def get_drive_download_url(file_url: str) -> str:
    match = re.search(r"/d/([^/]+)/", file_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return file_url


def save_google_drive(file_url):
    download_url = get_drive_download_url(file_url)
    print(f"download_url:{download_url}")

    output_dir = "documents"
    os.makedirs(output_dir, exist_ok=True)

    head = requests.get(download_url, stream=True)
    filename = None
    if "Content-Disposition" in head.headers:
        disposition = head.headers["Content-Disposition"]
        match = re.search(r'filename="(.+)"', disposition)
        if match:
            filename = match.group(1)

    if not filename:
        filename = f"google_drive_{os.urandom(4).hex()}.pdf"

    output_path = os.path.join(output_dir, filename)

    gdown.download(download_url, output_path, quiet=False)
    file_obj = None

    with open(output_path, "rb") as f:
        file_obj = ContentFile(f.read(), name=filename)

    return file_obj


# def get_onedrive_download_url(file_url: str) -> str:
#     """
#     Convert OneDrive sharing URL to direct download URL
#     """
#     try:
#         # Handle shortened OneDrive URLs (1drv.ms)
#         if "1drv.ms" in file_url:
#             # Resolve the shortened URL first
#             response = requests.head(file_url, allow_redirects=True)
#             file_url = response.url

#         # Method 1: Try to convert sharing URL to direct download
#         if "onedrive.live.com" in file_url:
#             # For modern OneDrive URLs, try different approaches

#             # Approach 1: Replace redir with download
#             if "redir?" in file_url:
#                 download_url = file_url.replace("redir?", "download?")
#                 if "&download=1" not in download_url:
#                     download_url += "&download=1"
#                 return download_url

#             # Approach 2: For embed URLs, convert to download
#             elif "embed?" in file_url:
#                 download_url = file_url.replace("embed?", "download?")
#                 if "&download=1" not in download_url:
#                     download_url += "&download=1"
#                 return download_url

#             # Approach 3: Add download parameter to existing URL
#             elif "?" in file_url:
#                 download_url = file_url + "&download=1"
#                 return download_url
#             else:
#                 download_url = file_url + "?download=1"
#                 return download_url

#         # Handle OneDrive for Business (SharePoint)
#         elif "sharepoint.com" in file_url or "-my.sharepoint.com" in file_url:
#             if "?" in file_url:
#                 download_url = file_url + "&download=1"
#             else:
#                 download_url = file_url + "?download=1"
#             return download_url

#         # If URL format is not recognized, try as-is
#         return file_url

#     except Exception as e:
#         print(f"Error processing OneDrive URL: {e}")
#         return file_url


# def save_onedrive_file(file_url):
#     """
#     Download OneDrive file and save it to local directory
#     Uses multiple strategies including Microsoft Graph API
#     """
#     try:
#         # Try multiple download strategies in order of likelihood to work
#         strategies = [
#             lambda url: get_microsoft_graph_url(url),  # Try Graph API first
#             lambda url: get_onedrive_download_url(url),  # Traditional method
#             lambda url: get_alternative_onedrive_url(url),  # Alternative methods
#             lambda url: url,  # Try original URL as last resort
#         ]

#         # Set up session with browser-like headers
#         session = requests.Session()
#         session.headers.update(
#             {
#                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
#                 "Accept": "application/octet-stream,*/*",
#                 "Accept-Language": "en-US,en;q=0.9",
#                 "Accept-Encoding": "gzip, deflate, br",
#                 "DNT": "1",
#                 "Connection": "keep-alive",
#                 "Sec-Fetch-Dest": "document",
#                 "Sec-Fetch-Mode": "navigate",
#                 "Sec-Fetch-Site": "cross-site",
#                 "Upgrade-Insecure-Requests": "1",
#             }
#         )

#         # Create output directory
#         output_dir = "documents"
#         os.makedirs(output_dir, exist_ok=True)

#         last_error = None

#         # Try each strategy
#         for i, strategy in enumerate(strategies):
#             try:
#                 download_url = strategy(file_url)
#                 print(f"Trying download strategy {i+1}: {download_url[:100]}...")

#                 # For Graph API, don't do HEAD request first
#                 if "graph.microsoft.com" in download_url:
#                     response = session.get(download_url, stream=True, timeout=60)
#                     response.raise_for_status()

#                     # Extract filename from original URL or response
#                     filename = (
#                         extract_filename_from_url(file_url)
#                         or f"onedrive_{os.urandom(4).hex()}.pdf"
#                     )
#                     output_path = os.path.join(output_dir, filename)

#                     # Save file
#                     save_file_with_progress(response, output_path, filename)
#                     return output_path
#                 else:
#                     # Test other URLs with HEAD request first
#                     head_response = session.head(
#                         download_url, allow_redirects=True, timeout=30
#                     )

#                     if head_response.status_code == 200:
#                         # Extract filename
#                         filename = extract_filename_from_response(
#                             head_response, file_url
#                         )
#                         output_path = os.path.join(output_dir, filename)

#                         # Download the file
#                         response = session.get(download_url, stream=True, timeout=60)
#                         response.raise_for_status()

#                         # Save file
#                         save_file_with_progress(response, output_path, filename)
#                         return output_path

#                     else:
#                         print(
#                             f"Strategy {i+1} failed with status code: {head_response.status_code}"
#                         )

#             except Exception as e:
#                 last_error = e
#                 print(f"Strategy {i+1} failed: {str(e)}")
#                 continue

#         # If all strategies fail, provide helpful error message
#         error_msg = f"All download strategies failed. This may be due to:\n"
#         error_msg += f"1. File requires authentication or login\n"
#         error_msg += f"2. File sharing permissions are restricted\n"
#         error_msg += f"3. File has been moved or deleted\n"
#         error_msg += f"4. OneDrive has changed their API policies\n"
#         error_msg += f"Last error: {last_error}"

#         raise Exception(error_msg)

#     except Exception as e:
#         raise Exception(f"Failed to download OneDrive file: {e}")


# def get_microsoft_graph_url(file_url: str) -> str:
#     """
#     Convert OneDrive sharing URL to Microsoft Graph API URL
#     This method works for public OneDrive shares without authentication
#     """
#     try:
#         import base64

#         # Clean the URL first
#         if "1drv.ms" in file_url:
#             # Resolve shortened URL
#             response = requests.head(file_url, allow_redirects=True, timeout=10)
#             file_url = response.url

#         # For OneDrive URLs, use Microsoft Graph API
#         if "onedrive.live.com" in file_url or "1drv.ms" in file_url:
#             # Encode the sharing URL for Microsoft Graph API
#             # Remove any existing download parameters
#             clean_url = file_url.split("&download=")[0].split("?download=")[0]

#             # Base64 encode the URL
#             encoded_url = base64.b64encode(clean_url.encode("utf-8")).decode("utf-8")
#             # Remove padding and make URL-safe
#             encoded_url = encoded_url.rstrip("=").replace("+", "-").replace("/", "_")

#             # Create Microsoft Graph API URL for shared items
#             graph_url = (
#                 f"https://graph.microsoft.com/v1.0/shares/u!{encoded_url}/root/content"
#             )
#             return graph_url

#         return file_url

#     except Exception as e:
#         print(f"Error creating Graph API URL: {e}")
#         return file_url


# def extract_filename_from_url(url):
#     """
#     Extract filename from OneDrive URL patterns
#     """
#     try:
#         # Try to extract from common OneDrive URL patterns
#         if "1drv.ms" in url:
#             # Pattern: https://1drv.ms/b/c/xyz/filename
#             parts = url.split("/")
#             if len(parts) > 6:
#                 return parts[-1].split("?")[0]

#         # Try URL parsing
#         from urllib.parse import urlparse

#         parsed = urlparse(url)
#         if parsed.path:
#             filename = os.path.basename(parsed.path)
#             if filename and filename != "/":
#                 return filename

#         return None

#     except Exception:
#         return None


# def get_alternative_onedrive_url(file_url: str) -> str:
#     """
#     Alternative method using Microsoft Graph API approach
#     """
#     try:
#         # Method 1: Try Microsoft Graph API approach for shared links
#         if "1drv.ms" in file_url or "onedrive.live.com" in file_url:
#             # Encode the sharing URL for Microsoft Graph API
#             import base64
#             from urllib.parse import quote

#             # Create Microsoft Graph API URL for shared items
#             encoded_url = base64.b64encode(file_url.encode("utf-8")).decode("utf-8")
#             # Remove padding for URL safety
#             encoded_url = encoded_url.rstrip("=")

#             # Create Graph API URL (this requires no authentication for public shares)
#             graph_url = (
#                 f"https://graph.microsoft.com/v1.0/shares/u!{encoded_url}/root/content"
#             )
#             return graph_url

#         # Method 2: For URLs that have been migrated to SharePoint
#         elif "migratedtospo=true" in file_url:
#             # Remove migration parameters and try SharePoint format
#             base_url = file_url.split("&migratedtospo=true")[0]
#             if "&download=1" not in base_url:
#                 return base_url + "&download=1"
#             return base_url

#         # Method 3: Try to extract the redeem parameter and construct new URL
#         elif "redeem=" in file_url:
#             redeem_param = file_url.split("redeem=")[1].split("&")[0]
#             try:
#                 # Decode base64 redeem parameter
#                 decoded_url = base64.b64decode(redeem_param + "==").decode("utf-8")
#                 if decoded_url.startswith("http"):
#                     return (
#                         decoded_url + "&download=1"
#                         if "?" in decoded_url
#                         else decoded_url + "?download=1"
#                     )
#             except:
#                 pass

#         return file_url

#     except Exception:
#         return file_url


# def extract_filename_from_response(response, original_url):
#     """
#     Extract filename from response headers or URL
#     """
#     filename = None

#     # Try to extract filename from Content-Disposition header
#     if "Content-Disposition" in response.headers:
#         disposition = response.headers["Content-Disposition"]
#         filename_match = re.search(r'filename[*]?=[\'"]?([^\'";]+)[\'"]?', disposition)
#         if filename_match:
#             filename = filename_match.group(1).strip()

#     # Fallback: try to extract filename from URL
#     if not filename:
#         parsed_url = urlparse(original_url)
#         url_filename = os.path.basename(parsed_url.path)
#         if url_filename and url_filename != "/":
#             filename = url_filename

#     # Final fallback: generate random filename
#     if not filename or filename == "/":
#         file_extension = ".pdf"  # Default extension
#         # Try to get extension from content-type
#         content_type = response.headers.get("Content-Type", "")
#         if "pdf" in content_type:
#             file_extension = ".pdf"
#         elif "word" in content_type or "document" in content_type:
#             file_extension = ".docx"
#         elif "excel" in content_type or "spreadsheet" in content_type:
#             file_extension = ".xlsx"
#         elif "powerpoint" in content_type or "presentation" in content_type:
#             file_extension = ".pptx"

#         filename = f"onedrive_{os.urandom(4).hex()}{file_extension}"

#     return filename


# def save_file_with_progress(response, output_path, filename):
#     """
#     Save file with progress tracking
#     """
#     total_size = int(response.headers.get("Content-Length", 0))
#     downloaded_size = 0

#     print(f"Downloading OneDrive file: {filename}")
#     if total_size > 0:
#         print(f"Total size: {total_size / (1024*1024):.2f} MB")

#     with open(output_path, "wb") as file:
#         for chunk in response.iter_content(chunk_size=8192):
#             if chunk:
#                 file.write(chunk)
#                 downloaded_size += len(chunk)

#                 # Show progress for large files
#                 if total_size > 1024 * 1024:  # Show progress for files > 1MB
#                     progress = (
#                         (downloaded_size / total_size) * 100 if total_size > 0 else 0
#                     )
#                     print(f"\rProgress: {progress:.1f}%", end="", flush=True)

#     if total_size > 1024 * 1024:
#         print()  # New line after progress

#     print(f"OneDrive file saved: {output_path}")
