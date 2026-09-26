import os.path
import io
from google.auth import credentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

FOLDER_ID = "1sl53d_l_k9zZHn2YV_vvTanfMAYNqNdL"


def get_latest_stat():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        #  service = build("drive", "v3", credentials=creds)

        #  # Call the Drive v3 API
        #  results = (
        #      service.files()
        #      .list(pageSize=10, fields="nextPageToken, files(id, name)")
        #      .execute()
        #  )
        #  items = results.get("files", [])

        #  if not items:
        #      print("No files found.")
        #      return
        #  print("Files:")
        #  for item in items:
        #      print(f"{item['name']} ({item['id']})")
        service = build("drive", "v3", credentials=creds)

        results = (
            service.files()
            .list(
                pageSize=1,
                corpora="user",
                q=f"'{FOLDER_ID}' in parents and trashed = false",
                orderBy="modifiedTime desc",
                fields="nextPageToken, files(id, name, capabilities/canDownload)",
            )
            .execute()
        )
        items = results.get("files", [])

        if not items:
            print("Smth wrong.")
            return
        print("Items:")
        for item in items:
            print(f"{item['name']} | {item['id']}")

        item = items[0]
        item_id = item["id"]
        print("Attempting download...")
        if item["capabilities"]["canDownload"] is False:
            print("Can't download this nigguh")
            return
        else:
            print("I can do it nigguh")
        request = service.files().get_media(fileId=item_id)

        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False

        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress()) * 100}.")
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f"An error occurred: {error}")
        file = None

    if file is None:
        return None

    file.seek(0)
    with open("stats.pdf", "wb") as f:
        f.write(file.read())

    return file.getvalue() if file is not None else None


if __name__ == "__main__":
    get_latest_stat()
