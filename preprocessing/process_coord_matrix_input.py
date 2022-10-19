import numpy as np
import pandas as pd
from shapely import geometry
import matplotlib.pyplot as plt
import multiprocessing
import csv
import copy
import warnings

warnings.filterwarnings("ignore")
# np.set_printoptions(threshold=np.inf)

def make_vertice_array(pos, dim, angle):
    vertices = []
    vertices.append((pos[0] - dim[1] / 2, pos[1] - dim[0] / 2))
    vertices.append((pos[0] + dim[1] / 2, pos[1] - dim[0] / 2))
    vertices.append((pos[0] + dim[1] / 2, pos[1] + dim[0] / 2))
    vertices.append((pos[0] - dim[1] / 2, pos[1] + dim[0] / 2))
    theta = np.deg2rad(angle)  # Convert angle to radians
    cosang, sinang = np.cos(theta), np.sin(theta)
    # find center point of Polygon to use as pivot
    n = len(vertices)
    cx = sum(p[0] for p in vertices) / n
    cy = sum(p[1] for p in vertices) / n
    new_points = []
    for p in vertices:
        x, y = p[0], p[1]
        tx, ty = x - cx, y - cy
        new_x = (tx * cosang + ty * sinang) + cx
        new_y = (-tx * sinang + ty * cosang) + cy
        new_points.append((new_x, new_y))
    return new_points


def add_buildings():
    """
    Add the buildings from Rosslyn
    """
    polygons = []
    # Rosslyn
    build_vertice = (
        (855, 402),
        (808, 403),
        (808, 445),
        (791, 443),
        (777, 448),
        (768, 644),
        (780, 657),
        (831, 657),
    )
    build_polygon = geometry.Polygon(build_vertice)
    polygons.append(build_polygon)
    build_vertice = (
        (816, 687),
        (710, 692),
        (708, 733),
        (693, 735),
        (691, 710),
        (660, 712),
        (660, 755),
        (764, 746),
    )
    build_polygon = geometry.Polygon(build_vertice)
    polygons.append(build_polygon)
    build_vertice = (
        (841, 295),
        (842, 317),
        (830, 340),
        (806, 354),
        (802, 371),
        (706, 433),
        (684, 435),
        (664, 424),
    )
    build_polygon = geometry.Polygon(build_vertice)
    polygons.append(build_polygon)
    build_vertice = (
        (658, 660),
        (685, 661),
        (685, 657),
        (720, 658),
        (722, 635),
        (735, 635),
        (751, 458),
        (740, 467),
        (700, 471),
        (700, 484),
        (660, 483),
    )
    build_polygon = geometry.Polygon(build_vertice)
    polygons.append(build_polygon)
    return polygons


def position_matrix_per_object_shape(bounds, resolution):
    bounds = np.array(bounds).reshape(2, 2)
    # shape each "image"
    shape = ((bounds[1] - bounds[0]) / resolution).astype(int)
    # tensorflow requires int
    shape = [int(i) for i in shape]
    return tuple(shape)


def _calc_position_matrix_row(args):
    (
        i,
        matrix,
        polygon_list,
        resolution,
        bounds,
        polygons_of_interest_idx_list,
        tx_target,
        polygon_z,
    ) = args
    for j in range(matrix.shape[2]):
        # create the point starting in (0, 0) with resolution "1"
        point_np = np.array((i, j))
        # apply the resolution and translate to the "area of interest"
        point_np = (point_np * resolution) + bounds[0]
        point = geometry.Point(point_np)
        for polygon_i, polygon in enumerate(polygon_list):
            if point.within(polygon):
                matrix[:, i, j] = 1 if polygon_z is None else polygon_z[polygon_i]
                if polygon_i in polygons_of_interest_idx_list:
                    polygon_idx = polygons_of_interest_idx_list.index(polygon_i)
                    matrix[polygon_idx, i, j] = 3
                elif polygon_i == tx_target:
                    matrix[:, i, j] = 5
    return matrix


def calc_position_matrix(
    bounds,
    polygon_list,
    resolution=1,
    polygons_of_interest_idx_list=None,
    tx_polygons_of_interest_idx_list=None,
    polygon_z=None,
):
    """Represents the receivers and other objects in a position matrix
    :param bounds: (minx, miny, maxx, maxy) of the region to study
    :param polygon_list: a list of polygons
    :param resolution: size of the pixel (same unity as bounds, default meters)
    :param polygons_of_interest_list: idx of polygon_list which will be "marked as receivers" on return, default: all
    :return: a matrix with shape (len(polygon_list), (maxx - minx) / resolution, (maxy - miny) / resolution)
    each point in the matrix[polygon_i] is a "pixel", the pixel is 1 when inside any polygon;
    2 when inside polygon_i; and 0 otherwise
    """
    if polygons_of_interest_idx_list is None:
        polygons_of_interest_idx_list = list(range(len(polygon_list)))
    bounds = np.array(bounds).reshape(2, 2)
    n_polygon = len(polygons_of_interest_idx_list)
    # shape each "image"
    shape = position_matrix_per_object_shape(bounds, resolution)
    # add n_polygon as first dimension
    shape = np.concatenate((np.array(n_polygon, ndmin=1), shape))
    # matrix = np.zeros(shape, dtype=np.uint8)
    tx_id = 0
    for tx in tx_polygons_of_interest_idx_list:
        matrix = np.zeros(shape, dtype=np.uint8)
        args = []
        for i in range(matrix.shape[1]):
            args.append(
                (
                    i,
                    matrix,
                    polygon_list,
                    resolution,
                    bounds,
                    polygons_of_interest_idx_list,
                    tx,
                    polygon_z,
                )
            )

        with multiprocessing.Pool() as pool:
            matrix_out = pool.map(_calc_position_matrix_row, args)

        for mat in matrix_out:
            matrix += mat
        if tx_id == 0:
            matrix_tx = matrix[np.newaxis, :, :, :]
        else:
            matrix_tx = np.vstack((matrix_tx, matrix[np.newaxis, :, :, :]))
        tx_id += 1

    return matrix_tx


def getInfoVehicles(sumo_info_file, analysis_area):
    ep = 0
    scene = 0
    # length,width,height in meters
    car_dim = [4.645, 1.775, 1.59]
    bus_dim = [12.5, 2.5, 4.3]
    truck_dim = [9, 2.4, 3.2]

    build_polygons = add_buildings()
    polygons = copy.deepcopy(build_polygons)

    polygons_of_interest_idx_list = []
    tx_polygons_of_interest_idx_list = []

    analysis_area_resolution = 1 # Resolution of the matrix image im meters

    with open(sumo_info_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            if ep != int(row["EpisodeID"]):
                # Create scene polygon
                print("Processing Episode {} and Scene {}".format(ep, scene))
                scene_position_matrix = calc_position_matrix(
                    analysis_area,
                    polygons,
                    analysis_area_resolution,
                    polygons_of_interest_idx_list,
                    tx_polygons_of_interest_idx_list,
                )
                # clean
                polygons = copy.deepcopy(build_polygons)
                polygons_of_interest_idx_list = []
                tx_polygons_of_interest_idx_list = []
                if (ep == 0 and scene==0):
                    output_position_matrix = scene_position_matrix[
                        np.newaxis, :, :, :, :
                    ]
                else:
                    # shouldRun = True
                    while True:
                        try:
                            output_position_matrix = np.vstack(
                                (
                                    output_position_matrix,
                                    scene_position_matrix[np.newaxis, :, :, :, :],
                                )
                            )
                            break
                        except ValueError:
                            if (
                                scene_position_matrix.shape[0]
                                != output_position_matrix.shape[1]
                            ):
                                fill_matrix = np.zeros_like(
                                    scene_position_matrix[0, :, :, :], dtype=float
                                )
                                fill_matrix = fill_matrix[np.newaxis, :, :, :]
                                fill_matrix.fill(np.nan)
                                scene_position_matrix = np.append(
                                    scene_position_matrix, fill_matrix, axis=0
                                )

                            elif (
                                scene_position_matrix.shape[1]
                                != output_position_matrix.shape[2]
                            ):
                                fill_matrix = np.zeros_like(
                                    scene_position_matrix[:, 0, :, :], dtype=float
                                )
                                fill_matrix = fill_matrix[:, np.newaxis, :, :]
                                fill_matrix.fill(np.nan)
                                scene_position_matrix = np.append(
                                    scene_position_matrix, fill_matrix, axis=1
                                )
                ep += 1
            position = [float(row["x"]), float(row["y"])]
            if row["TypeId"] == "Car":
                vertice = make_vertice_array(position, car_dim, float(row["angle"]))
            elif row["TypeId"] == "Bus":
                vertice = make_vertice_array(position, bus_dim, float(row["angle"]))
            elif row["TypeId"] == "Truck":
                vertice = make_vertice_array(position, truck_dim, float(row["angle"]))
            else:
                raise TypeError("Invalid Vehicle type")
            polygon = geometry.Polygon(vertice)
            polygons.append(polygon)
            if row["RxID"] != "-1":
                polygons_of_interest_idx_list.append(len(polygons) - 1)
            elif row["TxID"] != "-1":
                tx_polygons_of_interest_idx_list.append(len(polygons) - 1)

    scene_position_matrix = calc_position_matrix(
        analysis_area,
        polygons,
        analysis_area_resolution,
        polygons_of_interest_idx_list,
        tx_polygons_of_interest_idx_list,
    )
    # clean
    polygons = build_polygons
    polygons_of_interest_idx_list = []
    tx_polygons_of_interest_idx_list = []
    # Saves the last episode
    output_position_matrix = np.vstack(
        (output_position_matrix, scene_position_matrix[np.newaxis, :, :, :, :])
    )
    return output_position_matrix

if __name__ == "__main__":
    analysis_area = (743, 416, 771, 626) # Area of analyses from Rosslyn
    x1, y1, x2, y2 = analysis_area
    shape_of_image = (x2 - x1, y2 - y1)

    coord_path = "CoordVehiclesRxPerScene.csv"
    output_matrix = getInfoVehicles(coord_path, analysis_area)
    print(output_matrix.shape)
    outputFileName = 'coord_matrix_input' + '.npz'
    np.savez_compressed(
        outputFileName, 
        position_matrix_array=np.reshape(
            output_matrix, 
            [
                -1, 
                shape_of_image[0], 
                shape_of_image[1]
            ]
        )
    )
    print('==> Wrote file ' + outputFileName)
