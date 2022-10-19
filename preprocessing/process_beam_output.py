import numpy as np
import h5py
from preprocessing.mimo_channels import (
    getNarrowBandULAMIMOChannel,
    getDFTOperatedChannel,
)
import argparse
import json
import attrdict


def process_ep(path, cfg_file):
    with open(cfg_file, "r") as cfg_file:
        cfg = attrdict.AttrDict(json.load(cfg_file))
    number_Tx_antennas = cfg.DataGenerator.nTx
    number_Rx_antennas = cfg.DataGenerator.nRx
    normalizedAntDistance = 0.5
    angleWithArrayNormal = 0
    numOfInvalidChannels = 0
    beam_index = number_Rx_antennas * number_Tx_antennas
    h5_data = h5py.File(path)
    ray_data = np.array(h5_data.get("allEpisodeData"))
    numScenes = ray_data.shape[0]
    numTransmitters = ray_data.shape[1]
    numReceivers = ray_data.shape[2]
    channelOutputs = np.nan * np.ones(
        (
            numScenes,
            numTransmitters,
            numReceivers,
            number_Rx_antennas,
            number_Tx_antennas,
        ),
        np.complex128,
    )
    beamIndexOutputs = np.nan * np.ones(
        (numScenes, numTransmitters, numReceivers), np.float32
    )
    for s in range(numScenes):  # 50
        for t in range(numTransmitters):
            for r in range(numReceivers):  # 10
                insiteData = ray_data[s, t, r, :, :]
                numNaNsInThisChannel = sum(np.isnan(insiteData.flatten()))
                if numNaNsInThisChannel == np.prod(insiteData.shape):
                    numOfInvalidChannels += 1
                    continue  # next Tx / Rx pair
                if numNaNsInThisChannel > 0:
                    numMaxRays = insiteData.shape[0]
                    for itemp in range(numMaxRays):
                        if sum(np.isnan(insiteData[itemp].flatten())) > 0:
                            insiteData = insiteData[
                                itemp - 1, :
                            ]  # replace by smaller array without NaN
                            break
                gain_in_dB = insiteData[:, 0]
                timeOfArrival = insiteData[:, 1]
                AoD_el = insiteData[:, 2]
                AoD_az = insiteData[:, 3]
                AoA_el = insiteData[:, 4]
                AoA_az = insiteData[:, 5]
                isLOSperRay = insiteData[:, 6]
                pathPhases = insiteData[:, 7]
                mimoChannel = getNarrowBandULAMIMOChannel(
                    AoD_az,
                    AoA_az,
                    gain_in_dB,
                    number_Tx_antennas,
                    number_Rx_antennas,
                    normalizedAntDistance,
                    angleWithArrayNormal,
                )
                equivalentChannel = getDFTOperatedChannel(
                    mimoChannel, number_Tx_antennas, number_Rx_antennas
                )
                equivalentChannelMagnitude = np.abs(equivalentChannel)
                beamIndexOutputs[s, t, r] = np.argmax(
                    equivalentChannelMagnitude, axis=None
                )
                channelOutputs[s, t, r] = np.abs(equivalentChannel)
    return beamIndexOutputs, channelOutputs


if __name__ == "__main__":

    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", "-e", help="Number of episodes of the simulation", type=int, required=True
    )
    parser.add_argument(
        "--path_dataset", "-p", help="Path of the simulation output", type=str, required=True
    )
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, required=True
    )
    args = parser.parse_args()

    for ep in range(args.episodes):
        print("Episode # ", ep)
        beamIndex, channel = process_ep(
            "{}/rosslyn_ray_tracing_60GHz_e{}.hdf5".format(args.path_dataset, ep),
            args.cfg_file
        )
        if ep == 0:
            output_beam_matrix = beamIndex[np.newaxis, :, :, :]
            output_channel_matrix = channel[np.newaxis, :, :, :]
            continue
        output_beam_matrix = np.vstack(
            (output_beam_matrix, beamIndex[np.newaxis, :, :, :])
        )
        output_channel_matrix = np.vstack(
            (output_channel_matrix, channel[np.newaxis, :, :, :])
        )
    outputFileName = "beam_output" + ".npz"
    np.savez(
        outputFileName,
        beam_index_array=np.reshape(output_beam_matrix, -1),
    )
    print("==> Wrote file " + outputFileName)
