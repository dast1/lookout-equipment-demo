# Standard python and AWS imports:
import boto3
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import uuid

from botocore.config import Config
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from typing import List, Dict
from tqdm import tqdm


# Parameters
DEFAULT_REGION = 'eu-west-1'

def get_client(region_name=DEFAULT_REGION):
    """
    Get a boto3 client for the Amazon Lookout for Equipment service.
    
    PARAMS
    ======
        region_name: string
            AWS region name. (Default: eu-west-1)
    
    RETURN
    ======
        lookoutequipment_client
            A boto3 client to interact with the L4E service
    """
    lookoutequipment_client = boto3.client(
        service_name='lookoutequipment',
        region_name=region_name,
        config=Config(connect_timeout=30, read_timeout=30, retries={'max_attempts': 3}),
        endpoint_url=f'https://lookoutequipment.{region_name}.amazonaws.com/'
    )
    
    return lookoutequipment_client


def list_datasets(
    dataset_name_prefix=None,
    max_results=50,
    region_name=DEFAULT_REGION
):
    """
    List all the Lookout for Equipment datasets available in this account.
    
    PARAMS
    ======
        dataset_name_prefix: string
            Prefix to filter out all the datasets which names starts by 
            this prefix. Defaults to None to list all datasets.
            
        max_results: integer (default: 50)
            Max number of datasets to return 
            
        region_name: string
            AWS region name. (Default: eu-west-1)
            
    RETURN
    ======
        dataset_list: list of strings
            A list with all the dataset names found in the current region
    """
    # Initialization:
    dataset_list = []
    has_more_records = True
    lookoutequipment_client = get_client(region_name=region_name)
    
    # Building the request:
    kargs = {"MaxResults": max_results}
    if dataset_name_prefix is not None:
        kargs["DatasetNameBeginsWith"] = dataset_name_prefix
    
    # We query for the list of datasets, until there are none left to fetch:
    while has_more_records:
        # Query for the list of L4E datasets available for this AWS account:
        list_datasets_response = lookoutequipment_client.list_datasets(**kargs)
        if "NextToken" in list_datasets_response:
            kargs["NextToken"] = list_datasets_response["NextToken"]
        else:
            has_more_records = False
        
        # Add the dataset names to the list:
        dataset_summaries = list_datasets_response["DatasetSummaries"]
        for dataset_summary in dataset_summaries:
            dataset_list.append(dataset_summary['DatasetName'])
    
    return dataset_list

def list_models_for_datasets(
    model_name_prefix=None, 
    dataset_name_prefix=None,
    max_results=50,
    region_name=DEFAULT_REGION
):
    """
    List all the models available in a given region.
    
    PARAMS
    ======
        model_name_prefix: string (default: None)
            Prefix to filter on the model name to look for
            
        dataset_name_prefix: string (default None)
            Prefix to filter the dataset name: if used, only models
            making use of this particular dataset are returned

        max_results: integer (default: 50)
            Max number of datasets to return 
            
    RETURNS
    =======
        models_list: list of string
            List of all the models corresponding to the input parameters
            (regions and dataset)
    """
    # Initialization:
    models_list = []
    has_more_records = True
    lookoutequipment_client = get_client(region_name=region_name)
    
    # Building the request:
    list_models_request = {"MaxResults": max_results}
    if model_name_prefix is not None:
        list_models_request["ModelNameBeginsWith"] = model_name_prefix
    if dataset_name_prefix is not None:
        list_models_request["DatasetNameBeginsWith"] = dataset_name_prefix

    # We query for the list of models, until there are none left to fetch:
    while has_more_records:
        # Query for the list of L4E models available for this AWS account:
        list_models_response = lookoutequipment_client.list_models(**list_models_request)
        if "NextToken" in list_models_response:
            list_models_request["NextToken"] = list_models_response["NextToken"]
        else:
            has_more_records = False

        # Add the model names to the list:
        model_summaries = list_models_response["ModelSummaries"]
        for model_summary in model_summaries:
            models_list.append(model_summary['ModelName'])

    return models_list


def create_dataset(
    dataset_name, 
    dataset_schema, 
    region_name=DEFAULT_REGION
):
    """
    Creates a Lookout for Equipment dataset
    
    PARAMS
    ======
        dataset_name: string
            Name of the dataset to be created.
            
        dataset_schema: string
            JSON-formatted string describing the data schema the dataset
            must conform to.
            
        dataset_schema: string
            JSON formatted string to describe the dataset schema
    """
    # Initialization:
    lookoutequipment_client = get_client(region_name=region_name)
    has_more_records = True
    pp = pprint.PrettyPrinter(depth=4)

    # Checks if the dataset already exists:
    list_datasets_response = lookoutequipment_client.list_datasets(
        DatasetNameBeginsWith=dataset_name
    )

    dataset_exists = False
    for dataset_summary in list_datasets_response['DatasetSummaries']:
        if dataset_summary['DatasetName'] == dataset_name:
            dataset_exists = True
            break

    # If the dataset exists we just returns that message:
    if dataset_exists:
        print(f'Dataset "{dataset_name}" already exists and can be used to ingest data or train a model.')

    # Otherwise, we create it:
    else:
        print(f'Dataset "{dataset_name}" does not exist, creating it...\n')

        try:
            client_token = uuid.uuid4().hex
            data_schema = { 'InlineDataSchema': dataset_schema }
            create_dataset_response = lookoutequipment_client.create_dataset(
                DatasetName=dataset_name,
                DatasetSchema=data_schema,
                ClientToken=client_token
            )

            print("=====Response=====\n")
            pp.pprint(create_dataset_response)
            print("\n=====End of Response=====")

        except Exception as e:
            print(e)
            

def ingest_data(data_ingestion_role_arn, dataset_name, bucket, prefix, region_name=DEFAULT_REGION):
    lookoutequipment_client = get_client(region_name=region_name)
    ingestion_input_config = dict()
    ingestion_input_config['S3InputConfiguration'] = dict(
        [
            ('Bucket', bucket),
            ('Prefix', prefix)
        ]
    )

    client_token = uuid.uuid4().hex

    # Start data ingestion
    start_data_ingestion_job_response = lookoutequipment_client.start_data_ingestion_job(
        DatasetName=dataset_name,
        RoleArn=data_ingestion_role_arn, 
        IngestionInputConfiguration=ingestion_input_config,
        ClientToken=client_token)

    data_ingestion_job_id = start_data_ingestion_job_response['JobId']
    data_ingestion_status = start_data_ingestion_job_response['Status']
    
    return data_ingestion_job_id, data_ingestion_status


def delete_dataset(DATASET_NAME, region_name=DEFAULT_REGION):
    lookoutequipment_client = get_client(region_name=region_name)
    
    try:
        delete_dataset_response = lookoutequipment_client.delete_dataset(DatasetName=DATASET_NAME)
        print(f'Dataset "{DATASET_NAME}" is deleted successfully.')
        
    except Exception as e:
        error_code = e.response['Error']['Code']
        if (error_code == 'ConflictException'):
            print('Dataset is used by at least a model, deleting the associated model(s) before deleting dataset.')
            models_list = list_models_for_datasets(DATASET_NAME_FOR_LIST_MODELS=DATASET_NAME)

            for model_name_to_delete in models_list:
                delete_model_response = lookoutequipment_client.delete_model(ModelName=model_name_to_delete)
                print(f'- Model "{model_name_to_delete}" is deleted successfully.')
                
            delete_dataset_response = lookoutequipment_client.delete_dataset(DatasetName=DATASET_NAME)
            print(f'Dataset "{DATASET_NAME}" is deleted successfully.')

        elif (error_code == 'ResourceNotFoundException'):
            print(f'Dataset "{DATASET_NAME}" not found: creating a dataset with this name is possible.')

            
def create_data_schema(component_fields_map: Dict):
    return json.dumps(_create_data_schema_map(component_fields_map=component_fields_map))

def _create_data_schema_map(component_fields_map: Dict):
    data_schema = dict()
    component_schema_list = list()
    data_schema['Components'] = component_schema_list

    for component_name in component_fields_map:
        component_schema = _create_component_schema(component_name, component_fields_map[component_name])
        component_schema_list.append(component_schema)

    return data_schema

def _create_component_schema(component_name: str, field_names: List):
    component_schema = dict()
    component_schema['ComponentName'] = component_name
    col_list = []
    component_schema['Columns'] = col_list

    is_first_field = True
    for field_name in field_names:
        if is_first_field:
            ts_col = dict()
            ts_col['Name'] = field_name
            ts_col['Type'] = 'DATETIME'
            col_list.append(ts_col)
            is_first_field = False
        else:
            attr_col = dict()
            attr_col['Name'] = field_name
            attr_col['Type'] = 'DOUBLE'
            col_list.append(attr_col)
    return component_schema
    
def plot_timeseries(timeseries_df, tag_name, 
                    start=None, end=None, 
                    plot_rolling_avg=False, 
                    labels_df=None, 
                    predictions=None,
                    tag_split=None,
                    custom_grid=True,
                    fig_width=18,
                    prediction_titles=None
                   ):
    if start is None:
        start = timeseries_df.index.min()
    elif type(start) == str:
        start = pd.to_datetime(start)
        
    if end is None:
        end = timeseries_df.index.max()
    elif type(end) == str:
        end = pd.to_datetime(end)
        
    if (tag_split is not None) & (type(tag_split) == str):
        tag_split = pd.to_datetime(tag_split)

    # Prepare the figure:
    fig_height = 4
    height_ratios = [8]
    nb_plots = 1
    
    if labels_df is not None:
        fig_height += 1
        height_ratios += [1.5]
        nb_plots += 1
        
    if predictions is not None:
        if type(predictions) == pd.core.frame.DataFrame:
            fig_height += 1
            height_ratios += [1.5]
            nb_plots += 1
        elif type(predictions) == list:
            fig_height += 1 * len(predictions)
            height_ratios = height_ratios + [1.5] * len(predictions)
            nb_plots += len(predictions)
            
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nb_plots, 1, height_ratios=height_ratios, hspace=0.5)
    ax = []
    for i in range(nb_plots):
        ax.append(fig.add_subplot(gs[i]))
        
    # Plot the time series signal:
    data = timeseries_df[start:end].copy()
    if tag_split is not None:
        ax[0].plot(data.loc[start:tag_split, 'Value'], linewidth=0.5, alpha=0.5, label=f'{tag_name} - Training', color='tab:grey')
        ax[0].plot(data.loc[tag_split:end, 'Value'], linewidth=0.5, alpha=0.8, label=f'{tag_name} - Evaluation')
    else:
        ax[0].plot(data['Value'], linewidth=0.5, alpha=0.8, label=tag_name)
    ax[0].set_xlim(start, end)
    
    # Plot a daily rolling average:
    if plot_rolling_avg == True:
        daily_rolling_average = data['Value'].rolling(window=60*24).mean()
        ax[0].plot(data.index, daily_rolling_average, alpha=0.5, color='white', linewidth=3)
        ax[0].plot(data.index, daily_rolling_average, label='Daily rolling leverage', color='tab:red', linewidth=1)

    # Configure custom grid:
    ax_id = 0
    if custom_grid:
        date_format = DateFormatter("%Y-%m")
        major_ticks = np.arange(start, end, 3, dtype='datetime64[M]')
        minor_ticks = np.arange(start, end, 1, dtype='datetime64[M]')
        ax[ax_id].xaxis.set_major_formatter(date_format)
        ax[ax_id].set_xticks(major_ticks)
        ax[ax_id].set_xticks(minor_ticks, minor=True)
        ax[ax_id].grid(which='minor', axis='x', alpha=0.8)
        ax[ax_id].grid(which='major', axis='x', alpha=1.0, linewidth=2)
        ax[ax_id].xaxis.set_tick_params(rotation=30)

    # Add the labels on a second plot:
    if labels_df is not None:
        ax_id += 1
        label_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='1min')
        label_data = pd.DataFrame(index=label_index)
        label_data.loc[:, 'Label'] = 0.0

        for index, row in labels_df.iterrows():
            event_start = row['Start']
            event_end = row['End']
            label_data.loc[event_start:event_end, 'Label'] = 1.0
            
        ax[ax_id].plot(label_data['Label'], color='tab:green', linewidth=0.5)
        ax[ax_id].set_xlim(start, end)
        ax[ax_id].fill_between(label_index, y1=label_data['Label'], y2=0, alpha=0.1, color='tab:green', label='Real anomaly range (label)')
        ax[ax_id].axes.get_xaxis().set_ticks([])
        ax[ax_id].axes.get_yaxis().set_ticks([])
        ax[ax_id].set_xlabel('Anomaly ranges (labels)', fontsize=12)
        
    # Add the labels (anomaly range) on a 
    # third plot located below the main ones:
    if predictions is not None:
        pred_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='1min')
        pred_data = pd.DataFrame(index=pred_index)
        
        if type(predictions) == pd.core.frame.DataFrame:
            ax_id += 1
            pred_data.loc[:, 'prediction'] = 0.0

            for index, row in predictions.iterrows():
                event_start = row['start']
                event_end = row['end']
                pred_data.loc[event_start:event_end, 'prediction'] = 1.0

            ax[ax_id].plot(pred_data['prediction'], color='tab:red', linewidth=0.5)
            ax[ax_id].set_xlim(start, end)
            ax[ax_id].fill_between(pred_index, 
                             y1=pred_data['prediction'],
                             y2=0, 
                             alpha=0.1, 
                             color='tab:red')
            ax[ax_id].axes.get_xaxis().set_ticks([])
            ax[ax_id].axes.get_yaxis().set_ticks([])
            ax[ax_id].set_xlabel('Anomaly ranges (Prediction)', fontsize=12)
            
        elif type(predictions) == list:
            for prediction_index, p in enumerate(predictions):
                ax_id += 1
                pred_data.loc[:, 'prediction'] = 0.0

                for index, row in p.iterrows():
                    event_start = row['start']
                    event_end = row['end']
                    pred_data.loc[event_start:event_end, 'prediction'] = 1.0
                
                ax[ax_id].plot(pred_data['prediction'], color='tab:red', linewidth=0.5)
                ax[ax_id].set_xlim(start, end)
                ax[ax_id].fill_between(pred_index,
                                 y1=pred_data['prediction'],
                                 y2=0, 
                                 alpha=0.1, 
                                 color='tab:red')
                ax[ax_id].axes.get_xaxis().set_ticks([])
                ax[ax_id].axes.get_yaxis().set_ticks([])
                ax[ax_id].set_xlabel(prediction_titles[prediction_index], fontsize=12)
        
    # Show the plot with a legend:
    ax[0].legend(fontsize=10, loc='upper right', framealpha=0.4)
        
    return fig, ax