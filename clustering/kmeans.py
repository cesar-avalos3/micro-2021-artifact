# Naively choosing one from each group at random
def kmeans_clustering(dataFrame,principalComponents_dataFrame,starting_from=1, ending_at=20, column_variable='gpc__cycles_elapsed.avg',print_output=False):
    total_runtime = dataFrame[column_variable].sum()
    complete_groups = []
    random_per_K = []
    if(ending_at > len(dataFrame)):
        ending_at = len(dataFrame)+1
    results = {'random_choices_projections':[], 'random_choices_vectors':[], 'random_choices_names':[], 
               'random_choices_id':[], 'random_choices_speedups':[],'first_choices_projections':[],
               'center_choices_projections':[], 'center_choices_vectors':[], 'center_choices_names':[],
               'center_choices_id':[], 'center_choices_speedups':[], 'center_choices_errors': [],
               'complete_groups':[],
               'first_choices_vectors':[], 'first_choices_names':[], 'first_choices_id':[], 'group_count':[], 
               'total_runtime':total_runtime, 'group_number': [], 'errors': [], 'speedups': [], 'number_of_kernels': []}
    for i in range(starting_from, ending_at):
        random_choices = [] # Randomly select a kernel from the group
        random_choices_name = []
        random_choices_id = []
        mean_choices = []   # Select the kernel with the value closest to the mean
        max_choices = []    # Select the largest kernel value
        first_choices = []   # Selects the first chronological value
        first_choices_name = []
        first_choices_id = []
        center_choices_id = []
        closest_to_mean = []
        closest_to_mean_names = []
        group_count = []    # Number of elements inside the cluster i
        complete_groups_df = []
        k_means = KMeans(n_clusters=i, random_state=4).fit(principalComponents_dataFrame)
        dataFrame['Segments'] = k_means.labels_
        center_ids = cluster_centers(dataFrame,principalComponents_dataFrame,k_means)
        per_group_random = []
        for group in np.unique(k_means.labels_):
            temp_df = dataFrame.loc[dataFrame['Segments'] == group]
            #closest_to_mean.append(temp_df.loc[center_ids[group], column_variable])
            #closest_to_mean_names.append(temp_df.loc[center_ids[group], 'Kernel Name'])
            complete_groups_df.append(temp_df)
            temp_df.index = range(len(temp_df))
            #value_first = temp_df.loc[0,column_variable]
            first_choices.append(temp_df.loc[0,column_variable])
            first_choices_name.append(temp_df.loc[0,'Kernel Name'])
            first_choices_id.append(temp_df.loc[0,'ID'])
            #center_choices_id.append(center_ids[group])
            temp_df_sorted = temp_df.sort_values(column_variable)
            temp_df_sorted.index = range(len(temp_df_sorted))
            group_count.append(len(temp_df))
            random_vals = []
            random_names = []
            random_ids = []
            random_choice = random.randint(0,len(temp_df)-1)
            random_choice_name = ''
            for i in range(10):
                random_choice_ = random.randint(0,len(temp_df))
                random_choice_name = ''
                try:
                    random_vals.append(temp_df_sorted.loc[random_choice_,column_variable] * len(temp_df))
                    random_names.append(temp_df_sorted.loc[random_choice, 'Kernel Name'])
                    random_ids.append(random_choice_)
                except:
                    random_vals.append(temp_df_sorted.loc[0,column_variable] * len(temp_df))
                    random_names.append(temp_df_sorted.loc[0, 'Kernel Name'])
                    random_ids.append(random_choice)
            max_choice = temp_df_sorted.loc[len(temp_df)-1, column_variable]
            try:
                value_mean = temp_df_sorted.loc[int(len(temp_df)/2),column_variable]
            except:
                print(int(len(temp_df)/2))
            try:
                value_random = temp_df_sorted.loc[random_choice,column_variable]
                random_choice_name = temp_df_sorted.loc[random_choice, 'Kernel Name']
            except:
                print("Why would this happen?")
                print("Random Choice: "+str(random_choice))
                print("Length dataFrame: "+str(len(temp_df_sorted)))
                value_random = temp_df_sorted.loc[0,column_variable]
                random_choice_name = temp_df_sorted.loc[0,'Kernel Name']
            #value_random = 0
            random_choices.append(value_random)
            random_choices_id.append(random_choice)
            random_choices_name.append(random_choice_name)
            mean_choices.append(value_mean)
            max_choices.append(max_choice)
            per_group_random.append({'random_vals': random_vals, 'random_names': random_names, 'random_ids': random_ids})
        complete_groups.append(complete_groups_df)
        random_runtime = [random_choices[i] * group_count[i] for i in range(len(random_choices))]
        mean_runtime = [mean_choices[i] * group_count[i] for i in range(len(random_choices))]
        max_runtime = [max_choices[i] * group_count[i] for i in range(len(random_choices))]
        first_runtime = [first_choices[i] * group_count[i] for i in range(len(random_choices))]
        #closest_runtime = [closest_to_mean[i] * group_count[i] for i in range(len(random_choices))]
        random_per_K.append(per_group_random)
        if(print_output):
            print('For '+str(i)+ ' groups')
            print('----------------------------------------------')
            print(dataFrame[column_variable])
            print('The actual run time is: '+str(total_runtime))
            print('----------------- Projections ----------------')
            print(' ')
            print('The random value run time is: '+str(np.sum(random_runtime)))
            print('The error is '+ str(np.sum(random_runtime) / total_runtime ))
            print('The mean value run time is: '+str(np.sum(mean_runtime)))
            print('The error is '+ str(np.sum(mean_runtime) / total_runtime ))
            print('The max value run time is: '+str(np.sum(max_runtime)))
            print('The error is '+ str(np.sum(max_runtime) / total_runtime ))
            print('The first value run time is: '+str(np.sum(first_runtime)))
            print('The error is '+ str(np.sum(first_runtime) / total_runtime ))
            print('The speedup is '+ str(total_runtime / np.sum(first_choices)))
            print('The reduced number of kernels '+str(len(first_runtime)))
            print('The total number of kernels '+str(len(dataFrame)))
            print('The first choice vector is '+str(first_choices))
            print('The number of elements per group is '+str(group_count))
            print('The names of the first choices are '+str(first_choices_name))
            print('The kernel IDs of the first choices are '+str(first_choices_id))
        #print('Their product is '+str( [first_choices[i] * group_count[i] for i in range(len(first_choices))] ))
            print(' ')
        results['errors'].append(np.abs((np.sum(first_runtime)-total_runtime)) / total_runtime)
        results['speedups'].append(total_runtime / np.sum(first_choices))
        results['first_choices_vectors'].append(first_choices)
        results['first_choices_projections'].append(first_runtime)
        results['first_choices_names'].append(first_choices_name)
        results['first_choices_id'].append(first_choices_id)
        results['random_choices_vectors'].append(random_choices)
        results['random_choices_projections'].append(random_runtime)
        results['random_choices_names'].append(random_choices_name)
        results['random_choices_id'].append(random_choices_id)
        #results['center_choices_projections'].append(closest_runtime)
        #results['center_choices_id'].append(center_choices_id)
        #results['center_choices_vectors'].append(closest_to_mean)
        #results['center_choices_names'].append(closest_to_mean_names)
        #results['center_choices_errors'].append(np.abs((total_runtime - np.sum(closest_runtime))) / total_runtime)
        results['group_count'].append(group_count)
        results['group_number'].append(group)
        results['complete_groups'].append(complete_groups)
        results['number_of_kernels'].append(len(dataFrame))
        results['random_per_K'] = random_per_K
        #print(finalDf_sorted.loc[finalDf_sorted['Segments'] == group])
    return results
#kmeans_clustering(finalDf_sorted, principalComponents_Large)

# Returns the ID of the element closest to the group's centers
def cluster_centers(dataFrame, principalComponents_dataFrame, k_means, debug=False):
    centers_vector = []
    centers_ids = []
    clusters_centers = k_means.cluster_centers_
    temp_pca_df = dataFrame.copy()
    #labels = ['principal component '+str(x+1) for x in range(len(np.unique(k_means.labels_)))]
    for group in np.unique(k_means.labels_):
        temp_df = dataFrame.loc[dataFrame['Segments'] == group]
        temp_grouped_pca_df = temp_df.filter(regex='principal')
        original_indeces = temp_grouped_pca_df.index.values.tolist()
        temp_grouped_pca_df.index = range(len(temp_grouped_pca_df))
        minimum_distance = temp_grouped_pca_df.loc[0,:].values.tolist()
        minimum_index = original_indeces[0]
        #print(minimum_distance)
        if(debug):
            print('The clusters centers are: '+str(clusters_centers[group]))
            print('The first data point is : '+str(minimum_distance))
        minimum_distance = distance.euclidean(minimum_distance, clusters_centers[group])
        if(debug):
            print(minimum_distance)
        for i in range(len(temp_grouped_pca_df)):
            data_point = temp_grouped_pca_df.loc[i,:].values.tolist()
            temp_distance = distance.euclidean(data_point, clusters_centers[group])
            if(temp_distance < minimum_distance):
                minimum_distance = temp_distance
                minimum_index = original_indeces[i]
        if(debug):
            print(minimum_distance, minimum_index)
        centers_ids.append(minimum_index)
    if(debug):
        print('Returns from function \nNew iteration\n\n')
    return centers_ids

def point_closest_to_all(dataFrame, principalComponents_dataFrame, k_means, debug=False):
    closest_ids = []
    for group in np.unique(k_means.labels_):
        temp_df = dataFrame.loc[dataFrame['Segments'] == group]
        temp_grouped_pca_df = temp_df.filter(regex='principal')
        original_indeces = temp_grouped_pca_df.index.value.tolist()