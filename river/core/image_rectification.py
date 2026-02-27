from river.core.loading_data import (load_frame,load_gcps_img,load_dist)
from river.core.coordinate_transform import oblique_view_transformation_matrix

# Function for performing transformation
def transform(df_frames,gcp_cam,gcp_date,gcp_time):
    points = load_gcps_img(gcp_cam,gcp_date,gcp_time)
    dist = load_dist(gcp_cam)
    _, _, frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time)
    
    # Extract coordinates for transformation
    x1_pix, y1_pix = points['point1']
    x2_pix, y2_pix = points['point2']
    x3_pix, y3_pix = points['point3']
    x4_pix, y4_pix = points['point4']

    # Extract distances for transformation
    d12=dist[0]
    d23=dist[1]
    d34=dist[2]
    d41=dist[3]
    d13=dist[4]
    d24=dist[5]

    # Calculate transformation matrix
    transformation = oblique_view_transformation_matrix(
        x1_pix, y1_pix,
        x2_pix, y2_pix,
        x3_pix, y3_pix,
        x4_pix, y4_pix,
        d12,
        d23,
        d34,
        d41,
        d13,
        d24,
        image_path=frame_path,
    )

    return transformation