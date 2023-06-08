from google.cloud import storage
import os



class StorageHelpers():

  def __init__(self) -> None:
      self.CLOUD_STORAGE_BUCKET = os.environ.get("CLOUD_STORAGE_BUCKET")
      self.storage_client = storage.Client()
      self.asset_download_location = "tmp.jpeg"

  def download_asset_from_bucket(self, asset):
      """Download an image from the bucket."""
      
      bucket = self.storage_client.get_bucket(self.CLOUD_STORAGE_BUCKET)
      blob = bucket.blob(asset)
      blob.download_to_filename(self.asset_download_location)
      print('Blob {} downloaded to {}.'.format(
          blob.name,
          self.asset_download_location))
      return blob
  
  def upload_asset_to_bucket(self, asset, blob_name, content_type):
      """Upload an image to the bucket."""
      bucket = self.storage_client.get_bucket(self.CLOUD_STORAGE_BUCKET)
      blob = bucket.blob(blob_name)
      blob.cache_control = 'no-cache'
      if isinstance(asset, str):
        uploaded_asset_name = asset
        blob.upload_from_filename(asset, content_type=content_type)
      elif isinstance(asset, bytes):
        uploaded_asset_name = blob_name
        blob.upload_from_string(asset, content_type=content_type)
      else:
        uploaded_asset_name = asset.filename
        blob.upload_from_string(asset.read(), content_type=asset.content_type)
      blob.make_public()
      print('File {} uploaded to {} as {}.'.format(
          uploaded_asset_name,
          self.CLOUD_STORAGE_BUCKET,
          blob_name))
      return blob
