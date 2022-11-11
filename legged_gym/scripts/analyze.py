# create a python script that reads the .log file and plots the data
#
# the script should be able to plot any data in the .log file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import logging
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm



def main():
    # create a parser to read the arguments from the command line
    parser = argparse.ArgumentParser(description='Analyze the log file')
    parser.add_argument('--log', type=str, default='log.log', help='log file to be analyzed')
    parser.add_argument('--plot', type=str, default='none', help='plot the data')
    parser.add_argument('--save', type=str, default='no', help='save the plot')
    parser.add_argument('--show', type=str, default='yes', help='show the plot')
    parser.add_argument('--save_path', type=str, default='.', help='path to save the plot')
    parser.add_argument('--save_name', type=str, default='plot', help='name of the plot')
    parser.add_argument('--save_format', type=str, default='png', help='format of the plot')

    # parse the arguments
    args = parser.parse_args()

    # set the log file
    log_file = args.log
    
    # set the plot
    plot = args.plot

    # set the save
    save = args.save

    # set the show
    show = args.show

    # set the save path
    save_path = args.save_path

    # set the save name
    save_name = args.save_name

    # set the save format
    save_format = args.save_format
    
    with open(log_file, 'r') as f:
        header = f.readline().split(' ')
        header = [x for x in header if x != '']



    # Read the data from the log file where the values are separated by a || and do not read the last column
    data = pd.read_csv(log_file, sep=' ', header=None, skiprows=2, engine='python', names=header)

    

    # delete the \n column
    del data['\n']
    del header[-1]
    
    # Plot the data
    if plot == 'all':
        for i in range(len(header)):
            plt.figure()
            plt.plot(data[header[i]])
            plt.xlabel('time')
            plt.ylabel(header[i])
            plt.title(header[i])
            if save == 'yes':
                plt.savefig(save_path + '/' + save_name + '_' + header[i] + '.' + save_format)
            if show == 'yes':
                plt.show()
    # else:
    #     plt.figure()
    #     plt.plot(data[plot])
    #     plt.xlabel('time')
    #     plt.ylabel(plot)
    #     plt.title(plot)
    #     if save == 'yes':
    #         plt.savefig(save_path + '/' + save_name + '_' + plot + '.' + save_format)
    #     if show == 'yes':
    #         plt.show()
    
    # each leg is the entry in contact_forces_fx, contact_forces_fy, contact_forces_fz
    # each leg has 4 contact points
    # each contact point has 3 forces

    feet_order = ['FL', 'FR', 'RL', 'RR']
    data.to_csv('file.csv')
    # get the indices where the contact is true for each leg
    contact_indices = {}
    for leg in feet_order:
        contact_indices[leg] = data[data['contact_' + leg] == 1].index.values

    # for each leg, plot contact force fy vs contact force fz where the key should be the contact force plus the feet order, the plot should contain 4 subplots
    plt.figure()
    for i in range(4):
        # create a subplot for each leg
        plt.subplot(2, 2, i+1)
        # plot the contact force fy vs contact force fz for the contact indices of the leg
        # absolute value of the contact force fz
        plt.plot(np.abs(data['contact_forces_world_z_' + feet_order[i]][contact_indices[feet_order[i]]]), np.sqrt(np.abs(data['contact_forces_world_y_' + feet_order[i]][contact_indices[feet_order[i]]])**2 +  np.abs(data['contact_forces_world_x_' + feet_order[i]][contact_indices[feet_order[i]]])**2), '.')
        plt.xlabel('Fz')
        plt.ylabel('sqrt(Fx^2 + Fy^2)')
        plt.title("Leg name: " + feet_order[i])
        plt.suptitle('F_z leg frame vs F_x and F_y world frame')
        # create a subplot for each leg

        # plt.plot(data['contact_force_z_' + feet_order[i]],data['contact_force_y_' + feet_order[i]],  '.')
        # plt.xlabel('contact force z')
        # plt.ylabel('contact force y')
        # plt.title(feet_order[i])
    if save == 'yes':
        plt.savefig(save_path + '/' + save_name + '_' + feet_order[i] + '.' + save_format)
    if show == 'yes':
        plt.show()

    plt.figure()
    x = 1
    a = 10
    for i in range(4):
        plt.subplot(2, 2, i+1)
        # plot the ratio of the contact force fy vs contact force fz for the contact indices of the leg
        # print(data['contact_force_y_' + feet_order[i]][contact_indices[feet_order[i]]])
        temp_mat = np.sqrt(np.abs(data['contact_forces_world_y_' + feet_order[i]][contact_indices[feet_order[i]]])**2 +  np.abs(data['contact_forces_world_x_' + feet_order[i]][contact_indices[feet_order[i]]])**2)/np.abs(data['contact_force_z_' + feet_order[i]][contact_indices[feet_order[i]]])
        temp_mat[temp_mat > 2.0] = 2.0
        plt.plot(data['time'][contact_indices[feet_order[i]]],temp_mat, '.')

        # plt.plot(data['time'][contact_indices[feet_order[i]]],np.sqrt(np.abs(data['contact_forces_world_y_' + feet_order[i]][contact_indices[feet_order[i]]])**2 +  np.abs(data['contact_forces_world_x_' + feet_order[i]][contact_indices[feet_order[i]]])**2)/np.abs(data['contact_force_z_' + feet_order[i]][contact_indices[feet_order[i]]])/(1-np.exp(-x/a)), '.', color='black')
        plt.plot(data['time'],(data['robot_static_friction'] + data['ground_plane_static_friction'])/2, '-', color='red')
        plt.legend(['linear', 'exponential', 'ground truth average'])
        plt.xlabel('time')
        plt.ylabel('friction coefficient') 
        plt.title("Leg name: " + feet_order[i])
        # put a title for all the subplots
        plt.suptitle('Friction Coefficient vs Time where mu_s=' + str((data['robot_static_friction'][1] + data['ground_plane_static_friction'][1])/2) + ' and mu_k=' + str(data['ground_plane_dynamic_friction'][1]), fontsize=16)
    if save == 'yes':
        plt.savefig(save_path + '/' + save_name + '_' + feet_order[i] + '.' + save_format)
    if show == 'yes':
        plt.show()
    
    # plt.figure()
    # for i in range(4):
    #     # create a subplot for each leg
    #     plt.subplot(2, 2, i+1)
    #     # plot the contact force fy vs contact force fz for the contact indices of the leg
    #     plt.plot(np.abs(data['contact_forces_world_z_' + feet_order[i]][contact_indices[feet_order[i]]]), np.sqrt(np.abs(data['contact_forces_world_y_' + feet_order[i]][contact_indices[feet_order[i]]])**2 +  np.abs(data['contact_forces_world_x_' + feet_order[i]][contact_indices[feet_order[i]]])**2), '.')
    #     plt.xlabel('Fz')
    #     plt.ylabel('sqrt(Fx^2 + Fy^2)')
    #     plt.title("Leg name: " + feet_order[i])
    #     plt.suptitle('F_z world frame vs F_x and F_y world frame')
    #     # create a subplot for each leg

    #     # plt.plot(data['contact_force_z_' + feet_order[i]],data['contact_force_y_' + feet_order[i]],  '.')
    #     # plt.xlabel('contact force z')
    #     # plt.ylabel('contact force y')
    #     # plt.title(feet_order[i])
    # if save == 'yes':
    #     plt.savefig(save_path + '/' + save_name + '_' + feet_order[i] + '.' + save_format)
    # if show == 'yes':
    #     plt.show()

    # plt.figure()
    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     # plot the ratio of the contact force fy vs contact force fz for the contact indices of the leg
    #     temp_mat = np.abs(np.sqrt(np.abs(data['contact_forces_world_y_' + feet_order[i]][contact_indices[feet_order[i]]])**2 +  np.abs(data['contact_forces_world_x_' + feet_order[i]][contact_indices[feet_order[i]]])**2)/data['contact_forces_world_z_' + feet_order[i]][contact_indices[feet_order[i]]])
    #     # cap the temp_mat at 2.0
    #     temp_mat[temp_mat > 2.0] = 2.0
    #     plt.plot(data['time'][contact_indices[feet_order[i]]],temp_mat, '.')
        
    #     plt.plot(data['time'][contact_indices[feet_order[i]]],np.sqrt(np.abs(data['contact_forces_world_y_' + feet_order[i]][contact_indices[feet_order[i]]])**2 +  np.abs(data['contact_forces_world_x_' + feet_order[i]][contact_indices[feet_order[i]]])**2)/np.abs(data['contact_force_z_' + feet_order[i]][contact_indices[feet_order[i]]])/(1-np.exp(-x/a)), '.', color='black')
    #     plt.plot(data['time'],(data['robot_static_friction'] + data['ground_plane_static_friction'])/2, '-', color='red')
    #     plt.legend(['linear', 'exponential', 'ground truth average'])
    #     plt.xlabel('time')
    #     plt.ylabel('friction coefficient') 
    #     plt.title("Leg name: " + feet_order[i])
    #     plt.suptitle('Friction Coefficient vs Time where mu_s=' + str((data['robot_static_friction'][1] + data['ground_plane_static_friction'][1])/2) + ' and mu_k=' + str(data['ground_plane_dynamic_friction'][1]), fontsize=16)
    # if save == 'yes':
    #     plt.savefig(save_path + '/' + save_name + '_' + feet_order[i] + '.' + save_format)
    # if show == 'yes':
    #     plt.show()

    # get an array where the time is less than 10 seconds
    time_less_than_10 = data['time'] < 100
    # get an array where the time between 5 ND 10 seconds
    # time_less_than_10 = np.logical_and(data['time'] > 5, data['time'] < 10)
    # get the indices where the time is less than 10 seconds
    time_less_than_10_indices = np.where(time_less_than_10 == True)[0]

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        # plot the the contact force fz versus time for the contact indices of the leg
        plt.plot(data['time'][time_less_than_10_indices],data['contact_forces_world_x_' + feet_order[i]][time_less_than_10_indices], '.')
        plt.plot(data['time'][time_less_than_10_indices],data['contact_force_x_' + feet_order[i]][time_less_than_10_indices], '.')
        # plt.plot(data['time'][time_less_than_10_indices], np.sqrt(data['contact_force_y_' + feet_order[i]][time_less_than_10_indices]**2 + data['contact_force_x_' + feet_order[i]][time_less_than_10_indices]**2), '.')
        # plt.plot(data['time'][time_less_than_10_indices], np.sqrt(data['contact_forces_world_y_' + feet_order[i]][time_less_than_10_indices]**2 + data['contact_forces_world_x_' + feet_order[i]][time_less_than_10_indices]**2), '.', color='black')
        # plt.plot(data['contact_forces_world_z_' + feet_order[i]][time_less_than_10_indices], np.sqrt(data['contact_forces_world_y_' + feet_order[i]][time_less_than_10_indices]**2 + data['contact_forces_world_x_' + feet_order[i]][time_less_than_10_indices]**2), '.', color='black')
        plt.xlabel('time')
        plt.ylabel('sqrt(Fx^2 + Fy^2)')
        plt.title("Leg name: " + feet_order[i])
        plt.legend(['sensor force','contact force'])
    if save == 'yes':
        plt.savefig(save_path + '/' + save_name + '_' + feet_order[i] + '.' + save_format)
    if show == 'yes':
        plt.show()




if __name__ == '__main__':
    main()

    

