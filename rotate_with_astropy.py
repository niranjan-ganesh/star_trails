from PIL import Image, ImageChops
from astropy.coordinates import AltAz, EarthLocation, get_sun, SkyCoord
from astropy.time import Time
import numpy as np
import os


def get_pole_star_pixel_coords(image_path, location, observation_time):
    """
    Calculate the pixel coordinates of the Pole Star based on location and time.

    Args:
        image_path (str): Path to the image file.
        location (EarthLocation): Observer's location (latitude, longitude, and elevation).
        observation_time (str): Observation time in ISO 8601 format.

    Returns:
        tuple: Pixel coordinates (x, y) of the Pole Star.
    """
    # Altitude and Azimuth of Polaris
    polaris = SkyCoord.from_name("Polaris")
    time = Time(observation_time)
    altaz = AltAz(location=location, obstime=time)
    polaris_altaz = polaris.transform_to(altaz)

    # Convert celestial coordinates to image pixel coordinates
    # Assuming that the center of the image corresponds to zenith (Alt=90°)
    image = Image.open(image_path)
    width, height = image.size
    center_x, center_y = width / 2, height / 2

    # Map Alt/Az to pixel coordinates
    scale = min(width, height) / 180  # Scaling factor (e.g., degrees to pixels)
    x_offset = scale * polaris_altaz.az.degree
    y_offset = scale * (90 - polaris_altaz.alt.degree)

    return int(center_x + x_offset), int(center_y - y_offset)


def create_star_trails(image_path, output_path, pivot_point, step_rotation=0.1, total_rotation=360):
    base_image = Image.open(image_path).convert("RGBA")
    width, height = base_image.size

    # Create a blank image for stacking
    stacked_image = Image.new("RGBA", base_image.size, (0, 0, 0, 255))

    # Create star trails by rotating and stacking
    num_layers = int(total_rotation / step_rotation)
    print(f"Generating {num_layers} layers with a step rotation of {step_rotation} degrees...")

    for i in range(num_layers):
        # Rotate the image
        angle = (i + 1) * step_rotation
        rotated_image = base_image.rotate(angle, resample=Image.BICUBIC, center=pivot_point)

        # Blend using lighten mode
        stacked_image = ImageChops.lighter(stacked_image, rotated_image)

        # Merge layers periodically to save memory
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{num_layers} layers merged.")

    # Save the final image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stacked_image.save(output_path, "PNG")
    print(f"Star trail image saved at {output_path}.")


# Parameters
input_image_path = "background_image.png"
output_image_path = "output/star_trails_with_pole_star.png"
observer_location = EarthLocation(lat=44.5, lon=-69.0,
                                  height=30)  # Example location (latitude, longitude, elevation in meters)
observation_time = "2024-12-01T22:00:00"  # Example observation time in ISO 8601 format

# Get pivot point
pivot_point = get_pole_star_pixel_coords(input_image_path, observer_location, observation_time)

# Generate star trails
create_star_trails(input_image_path, output_image_path, pivot_point, step_rotation=0.004178, total_rotation=2)