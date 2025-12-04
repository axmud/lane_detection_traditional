import carla
import random
import time
import numpy as np
import cv2


weather_options = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.WetCloudySunset,
        carla.WeatherParameters.MidRainSunset,
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.SoftRainSunset]
tm_port = 8000
sprawn_vehicles = []
sprawn_vehicles_num = 50
IMG_HEIGHT = 600
IMG_WIDTH =800
relative_transform = carla.Transform(carla.Location(x=1.5, z=1.7),carla.Rotation(pitch=0))
latest_image = {"data": None}
weather_index = 0
maps_list = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"]
map_index = 0


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    else:
        raise ValueError("orient must be 'x' or 'y'")

    max_val = np.max(abs_sobel)
    if max_val == 0:
        scaled_sobel = np.zeros_like(abs_sobel, dtype=np.uint8)
    else:
        scaled_sobel = np.uint8(255 * abs_sobel / max_val)

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    max_val = np.max(gradmag)
    if max_val == 0:
        scaled = np.zeros_like(gradmag, dtype=np.uint8)
    else:
        scale_factor = max_val / 255
        scaled = (gradmag / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # use arctan2 to be safer
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255), lthresh=(0, 255)):
    # HLS for S-channel (good for yellow/white lanes)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    # HSV for V-channel (overall brightness)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output


def preprocess_image(img_rgb):
    """
    img_rgb: image in RGB
    returns: binary mask (0/1) highlighting lane pixels
    """

    # Tune these thresholds later if needed
    gradx = abs_sobel_thresh(img_rgb, orient='x', sobel_kernel=3, thresh=(20, 255))
    grady = abs_sobel_thresh(img_rgb, orient='y', sobel_kernel=3, thresh=(20, 255))
    mag_binary = mag_thresh(img_rgb, sobel_kernel=3, mag_thresh=(30, 255))
    dir_binary = dir_threshold(img_rgb, sobel_kernel=15, thresh=(0.7, 1.3))

    color_binary = color_threshold(
        img_rgb,
        sthresh=(170, 255),   # strong S channel
        vthresh=(50, 255)
    )

    combined = np.zeros_like(dir_binary, dtype=np.uint8)

    # Combine: (gradients) OR (color)
    combined[(((gradx == 1) & (grady == 1)) | (mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 1

    return combined


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    classic sliding window to get lane pixels
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = int(binary_warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Window boundaries in y
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # Window boundaries in x
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter next window if enough pixels found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find lane pixels
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each
    left_fit = None
    right_fit = None
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    left_fitx = None
    right_fitx = None
    if left_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    if right_fit is not None:
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def detect_lanes_traditional(image_np):
    """
    Input:
        image_np - numpy array image (BGR or RGB)
    Output:
        result - lane-overlay image (BGR)
    """

    # -------------------------
    # 1. Prepare image
    # -------------------------
    # ensure BGR → RGB
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        raise ValueError("Input image must be HxWx3 array")

    img_bgr = image_np.copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = img_rgb
    img_size = (img.shape[1], img.shape[0])

    # -------------------------
    # 2. Perspective transform setup
    # -------------------------
    bot_width = .75
    mid_width = .1
    height_pct = .625
    bottom_trim = .935

    src = np.float32([
        [img.shape[1] * (.5 - mid_width / 2), img.shape[0] * height_pct],
        [img.shape[1] * (.5 + mid_width / 2), img.shape[0] * height_pct],
        [img.shape[1] * (.5 + bot_width / 2), img.shape[0] * bottom_trim],
        [img.shape[1] * (.5 - bot_width / 2), img.shape[0] * bottom_trim]
    ])

    offset = img_size[0] * .25
    dst = np.float32([
        [offset, 0],
        [img_size[0] - offset, 0],
        [img_size[0] - offset, img_size[1]],
        [offset, img_size[1]]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # -------------------------
    # 3. Preprocess → binary mask
    # -------------------------
    binary = preprocess_image(img)
    binary = (binary * 255).astype(np.uint8)

    # -------------------------
    # 4. Warp binary image
    # -------------------------
    warped = cv2.warpPerspective(binary, M, img_size, flags=cv2.INTER_LINEAR)

    # -------------------------
    # 5. Detect lane pixels
    # -------------------------
    left_fitx, right_fitx, ploty = fit_polynomial(warped)

    # -------------------------
    # 6. Draw lane overlay
    # -------------------------
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if left_fitx is not None and right_fitx is not None:
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # -------------------------
    # 7. Warp overlay back to original view
    # -------------------------
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)

    # -------------------------
    # 8. Add overlay onto original image
    # -------------------------
    result = cv2.addWeighted(img_bgr, 1, newwarp, 0.3, 0)

    return result


def load_new_world(client, town_name):
    new_world = client.load_world(town_name)
    settings = new_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    new_world.apply_settings(settings)
    return new_world


def destroy_vehicles():
    global sprawn_vehicles
    try:
        for vehicle in list(sprawn_vehicles):  # iterate over a copy
            try:
                vehicle.destroy()
            except Exception as e:
                print("Error destroying vehicle:", e)
        sprawn_vehicles.clear()
    except Exception as e:
        print("error in destroy_vehicles: ", e)
    print("Vehicles are destroyed")


def destroy_camera(camera):
    try:
        # stop stream & detach callback
        camera.stop()
        camera.listen(lambda image: None)  # clear callback
        time.sleep(0.1)  # small delay to let C++ side flush
    except Exception as e:
        print("Error stopping camera:", e)
    try:
        camera.destroy()
    except Exception as e:
        print("Error destroying camera:", e)
    print("Camera is destroyed")


def change_map(camera_rgb):
    """
    Destroy current actors, load next map, re-create everything.
    Returns: new_world, new_camera_rgb, new_ego_vehicle
    """
    global maps_list, map_index

    destroy_camera(camera_rgb)
    destroy_vehicles()

    new_client = connect_carla_and_create_client()

    print(f"Loading new world: {maps_list[map_index]}")
    new_world = load_new_world(new_client, maps_list[map_index])

    set_weather(new_world)

    new_tm = create_traffic_manager(new_client)

    create_sprawn_vehicles(new_world)

    new_ego_vehicle = create_ego_vehicle(new_world)

    new_camera_rgb = create_rgb_camera(new_world, new_ego_vehicle)

    listen_rgb_camera(new_camera_rgb)

    print(f"Map: {maps_list[map_index]}")

    map_index = map_index + 1
    if map_index == len(maps_list):
        map_index = 0

    return new_world, new_camera_rgb, new_ego_vehicle


def change_weather(world):
    global weather_index, weather_options
    world.set_weather(weather_options[weather_index])
    print("Weather changed to {}".format(weather_index))
    weather_index += 1
    if weather_index >= len(weather_options):
        weather_index = 0


def show_camera_image(image):
    global latest_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr = array[:, :, :3].copy()
    latest_image["data"] = bgr


def listen_rgb_camera(camera_rgb):
    camera_rgb.listen(lambda image: show_camera_image(image))


def warmup_world(world):
    for _ in range(5):
        world.tick()
        time.sleep(0.01)


def create_rgb_camera(world, vehicle):
    global IMG_HEIGHT, IMG_WIDTH
    blueprint_library = world.get_blueprint_library()
    cmr_rgb = blueprint_library.find('sensor.camera.rgb')
    cmr_rgb.set_attribute('role_name', 'camera')
    cmr_rgb.set_attribute('image_size_x', str(IMG_WIDTH))
    cmr_rgb.set_attribute('image_size_y', str(IMG_HEIGHT))
    cmr_rgb.set_attribute('fov', '90')
    camera_rgb = world.spawn_actor(cmr_rgb, relative_transform, attach_to=vehicle)
    return camera_rgb


def create_ego_vehicle(world, autopilot=True):
    global sprawn_vehicles, tm_port
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    while True:
        try:
            vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
            if vehicle:
                sprawn_vehicles.append(vehicle)
                print("Ego vehicle created")
                break
        except Exception as e:
            print(f"There is an error while creating ego vehicle :{e}\nTrying again...", flush=True)
    if autopilot:
        vehicle.set_autopilot(True, tm_port)
    return vehicle


def create_sprawn_vehicles(world):
    global sprawn_vehicles, tm_port
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for _ in range(sprawn_vehicles_num):
        try:
            vehicle = world.spawn_actor(random.choice(vehicle_bp), random.choice(spawn_points))
            if vehicle:
                sprawn_vehicles.append(vehicle)
        except RuntimeError:
            continue
    for vehicle in sprawn_vehicles:
        vehicle.set_autopilot(True, tm_port)
    print(f"{len(sprawn_vehicles)} vehicles spawned.")


def create_traffic_manager(client):
    global tm_port
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.global_percentage_speed_difference(0)
    return tm


def set_weather(world):
    """
    Randomly choose one of your preferred weather presets.
    """
    global weather_options
    weather = random.randint(0, len(weather_options) - 1)
    world.set_weather(weather_options[14])
    print("Weather set to {}".format(weather))


def create_world(client):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS sim
    world.apply_settings(settings)
    return world


def connect_carla_and_create_client():
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    return client


def main():
    client = connect_carla_and_create_client()

    world = create_world(client)

    set_weather(world)

    tm = create_traffic_manager(client)

    create_sprawn_vehicles(world)

    ego_vehicle = create_ego_vehicle(world)

    camera_rgb = create_rgb_camera(world, ego_vehicle)

    try:
        warmup_world(world)

        listen_rgb_camera(camera_rgb)

        print("Starting visualization loop.")
        print("q / ESC  : quit")
        print("e        : change weather")
        print("m        : change map")

        while True:
            world.tick()

            frame = latest_image["data"]

            if frame is not None:
                frame = detect_lanes_traditional(frame)
                cv2.imshow("Lane detection (autopilot, q/ESC to quit, w weather, m map)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: break
            elif key == ord('e'): change_weather(world)
            elif key == ord('m'):
                world, camera_rgb, ego_vehicle = change_map(camera_rgb)



    except Exception as e:
        print("Error in Main try-except block: ",e)
    finally:
        destroy_camera(camera_rgb)
        destroy_vehicles()
        cv2.destroyAllWindows()

        print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
