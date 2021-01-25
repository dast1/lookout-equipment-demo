import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import utils

from datetime import datetime, timedelta
from tqdm import tqdm

class HistorianData:
    def __init__(self, 
                 raw_data_root_dir, 
                 target_data_root_dir, 
                 verbose=1,
                 sampling_rate=None,
                 input_format='stream',
                 output_format='CSV'
                ):
        """
        HistorianData management class constructor.
        
        PARAMS
        ======
            raw_data_root_dir (string)
                Location of the raw data to manage
                
            target_data_root_dir (string)
                Location of the target processed data
                
            verbose (integer)
                Verbosity of the outputs (0 for no message, and 1 for verbose, default: 1)
                
            sampling_rate (timedelta)
                Sampling rate at which the signals must be considered
                
            input_format (string)
                Can be either "stream" or "tabular".
                - stream: input file only includes 3-4 fields: timestamp, tag 
                          name and quantity. Sometimes, also includes an error 
                          code
                - tabular: input file include timestamp and one column per tag
                
            output_format (string):
                Can be either CSV, HDF5 or PARQUET. Defaults to CSV.
        """
        self.raw_data_schema = None
        self.tags_dataframe = None
        self.tags_list = None
        self.all_tags_fname = None
        self.label_ranges = None
        self.ignored_sequences_df = None
        self.off_timestamps = None
        self.components_df = None
        self.timeseries_saved = False
        
        self.verbose = verbose
        self.sampling_rate = sampling_rate
        self.input_format = input_format
        self.output_format = output_format
        
        if os.path.exists(raw_data_root_dir):
            self.raw_data_root_dir = raw_data_root_dir
        else:
            raise Exception(f'Raw data root dir ({raw_data_root_dir}) does not exist')

        if os.path.exists(target_data_root_dir):
            self.target_data_root_dir = target_data_root_dir
        else:
            raise Exception(f'Target data root dir ({target_data_root_dir}) does not exist')
            
    def set_default_output_format(self, output_format='CSV'):
        self.output_format = output_format

    def define_raw_data_schema(self, schema):
        """
        Use this method to define how the raw data are structured using a Python 
        dictionary. Mandatory fields that must be found in the source raw data are
        timestamp, tag name, tag value and a state used to identify if a given value
        is valid or not.
        
        PARAMS
        ======
            schema (dict)
                A dictionary matching timestamp, tag, value and states.
                Example:
                    schema={
                        'timestamp': 'fx_date_time',
                        'tag': 'co_tag',
                        'value': 'qt_value',
                        'state': 'id_state'
                    }
            
        EXCEPTIONS
        ==========
            General exception is raised with an invalid schema: this exception is raised if the
            dictionary does not contain all the required fields.
        """
        if self.raw_data_schema is not None:
            print('Schema already defined:')
            print(self.raw_data_schema)
            return
            
        # Validate schema:
        if self.input_format == 'stream':
            expected_keys = ['timestamp', 'tag', 'value', 'state']
        elif self.input_format == 'tabular':
            expected_keys = ['timestamp']
            
        valid_schema = True
        for expected_key in expected_keys:
            if not expected_key in list(schema.keys()):
                valid_schema = False

        # Store the schema if it's a valid one:
        if valid_schema:
            self.raw_data_schema = schema
            
        # Raise an exception otherwise:
        else:
            self.raw_data_schema = None
            if self.input_format == 'stream':
                raise Exception('Invalid schema: schema must contain timestamp, tag, value and state')
            elif self.input_format == 'tabular':
                raise Exception('Invalid schema: schema must contain a timestamp')
            
    def delete_tags_dataframe(self, delete_schema=False):
        """
        Removes the tags dataframe from this object, allows loading a new
        one with either the same schema (if delete_schema is False) or with
        a new schema (if we also want to clean the schema).
        
        PARAMS
        ======
            delete_schema (boolean)
                If True, the function will also reset the data schema to allow
                you to load a new one. Defaults to False: schema is then left
                untouched to permit loading a new dataset with the existing
                data schema.
        """
        self.tags_dataframe = None
        self.tags_list = None
        if delete_schema:
            self.raw_data_schema = None
            
    def build_tags_dataframe(self, valid_states=[0], timestamp_format=None):
        """
        Loads the historian data from disk into a dataframe stored in the tags_dataframe 
        property. We filter out the invalid data (marked by the state field), drop all 
        duplicate values and sort by the timestamp field.
        
        PARAMS
        ======
            valid_states (list) - list of valid values for the state field to mark a given row as valid.
            
        EXCEPTIONS
        ==========
            General exceptions are raised:
            - If the schema is not defined for the raw data (using the define_raw_data_schema() method)
            - If no CSV file is found in the root directory.
        """
        if self.tags_dataframe is not None:
            if self.verbose == 1:
                print('Tags data already loaded from disk into the tags_dataframe property')
            return
            
        if self.raw_data_schema is None:
            raise Exception('Undefined schema, run define_raw_data_schema() before building the tags dataset.')
            
        # Walk the provided root directory: this directory should only contain CSV files with the same schema:
        df_list = []
        timestamp = self.raw_data_schema['timestamp']
        for root, dirs, files in os.walk(self.raw_data_root_dir):
            for f in files:
                if f[-3:] != 'csv':
                    if self.verbose == 1: print(f'Ignoring file {f} (does not appear to be a CSV file)')

                else:
                    filepath = os.path.join(self.raw_data_root_dir, f)

                    if self.verbose == 1:
                        size = utils.get_readable_filesize(os.stat(filepath).st_size)
                        print(f'Processing file {filepath} ({size})...')

                    current_df = pd.read_csv(filepath)
                    
                    if ('tag' in self.raw_data_schema) and (self.raw_data_schema['tag'] == ''):
                        tag_name = f[:-4]
                        current_df['tag'] = tag_name

                    # We only keep the datapoint that are not marked as erroneous and remove any duplicates:
                    if ('state' in self.raw_data_schema) and (self.raw_data_schema['state'] != ''):
                        current_df = current_df[current_df[self.raw_data_schema['state']].isin(valid_states)]
                        
                    current_df = current_df.drop_duplicates()
                    df_list.append(current_df)
                    del current_df
                    
        if ('tag' in self.raw_data_schema) and (self.raw_data_schema['tag'] == ''):
            self.raw_data_schema['tag'] = 'tag'
            
        if len(df_list) > 0:
            if len(df_list) == 1:
                self.tags_dataframe = df_list[0]
            else:
                if self.verbose == 1: print('Concatenating all the individual dataframes...')
                self.tags_dataframe = pd.concat(df_list, axis='index')
            
            if timestamp_format is not None:
                if self.verbose == 1: print('Converting timestamp format...')
                try:
                    if timestamp_format in ['D', 's', 'ms', 'us', 'ns']:
                        self.tags_dataframe[timestamp] = pd.to_datetime(self.tags_dataframe[timestamp], 
                                                                        unit=timestamp_format)
                    elif timestamp_format == 'None':
                        self.tags_dataframe[timestamp] = pd.to_datetime(self.tags_dataframe[timestamp], 
                                                                        format=None)
                    else:
                        self.tags_dataframe[timestamp] = pd.to_datetime(self.tags_dataframe[timestamp], 
                                                                        format=timestamp_format)
                    
                except Exception as e:
                    print(e)
            
            if self.verbose == 1: print('Sorting along the timestamp dimension...')
            self.tags_dataframe = self.tags_dataframe.sort_values(by=timestamp, ascending=True)
            self.tags_dataframe = self.tags_dataframe.reset_index(drop=True)
            
            if self.verbose == 1: print('Done.')
            for df in df_list: del df
            del df_list
            
        else:
            self.tags_dataframe = None
            raise Exception(f'No CSV files found in the root directory ({self.raw_data_root_dir})')
            
    def load_tags_dataframe(self, tags_fname):
        """
        Loads a tag dataframe from disk.
        
        PARAMS
        ======
            tags_fname (string)
                Filename containing a previously saved tags dataframe. We will
                look into the target_data_root_dir location.
                
        EXCEPTIONS
        ==========
            General exception raised if file is not found.
        """
        if self.tags_dataframe is not None:
            if self.verbose == 1:
                print('Tags data already loaded from disk into the tags_dataframe property')
            return
        
        tags_fname = os.path.join(self.target_data_root_dir, tags_fname)
        if not os.path.exists(tags_fname):
            raise Exception(f'Tags file not found in the following path: {tags_fname}')
            
        if self.verbose == 1: print('Tags file found, loading it...')
        extension = tags_fname.split('.')[-1]
        if extension == 'csv':
            self.tags_dataframe = pd.read_csv(tags_fname)
        elif (extension == 'h5') | (extension == 'hdf5'):
            self.tags_dataframe = pd.read_hdf(tags_fname, key='timeseries', mode='r')
        elif extension == 'parquet':
            table = pq.read_table(tags_fname)
            self.tags_dataframe = table.to_pandas()
            del table
            
        self.all_tags_fname = tags_fname
        if self.verbose == 1: print('Done.')
            
    def get_extension(self, output_format):
        if output_format == 'CSV':
            extension = '.csv'
        elif (output_format == 'HDF5') | (output_format == 'H5'):
            extension = '.h5'
        elif (output_format == 'PARQUET'):
            extension = '.parquet'
            
        return extension
            
    def save_tags_dataframe(self, 
                            tags_fname=None, 
                            overwrite=False, 
                            output_format='CSV'):
        """
        Persists the tags_dataframe property to disk.
        
        PARAMS
        ======
            tags_fname (string)
                Filename to use to save the tags dataframe. If None is 
                specified, will create a unique filename called 
                all-tags-yyyymmdd-hhmmss.csv
                
            overwrite (boolean)
                Specifies if any existing file should be overwritten 
                if found (defaults to False)
                
            output_format (string)
                CSV to save to .csv format (default behavior), HDF5 
                (saved as .h5 file) or PARQUET (saved as .parquet file)
        
        EXCEPTIONS
        ==========
            General exceptions are raised:
            - If no dataset has been loaded with the build_tags_dataframe() method
            - If the target file already exists and overwrite is set to False
        """
        if self.tags_dataframe is None:
            raise Exception('No dataset loaded. Use the build_tags_dataframe() method to load a dataset from disk.')
            
        # If a file was previously saved, we reuse the same filename:
        if self.all_tags_fname is not None:
            tags_fname = self.all_tags_fname.split('/')[-1]
            if self.verbose == 1:
                print(f'Reusing this target location: {self.all_tags_fname}')
                
        extension = self.get_extension(output_format)
                
        # Generates a filename if None is specified
        # for the tags_fname argument:
        if tags_fname is None:
            uid = datetime.now().strftime("%Y%m%d-%H%M%S")
            tags_fname = os.path.join(self.target_data_root_dir, f'all-tags-{uid}{extension}')
            if self.verbose == 1:
                print(f'No target filename given, saving file to {tags_fname}')
                
        else:
            tags_fname = os.path.join(self.target_data_root_dir, tags_fname)
            if self.verbose == 1:
                print(f'Saving file to {tags_fname}')
                
        if (os.path.exists(tags_fname)) & (overwrite == False):
            raise Exception('File exists, use overwrite=True if you want to overwrite it')

        if self.verbose == 1: print(f'Saving file to disk...')
        if output_format == 'CSV':
            self.tags_dataframe.to_csv(tags_fname, index=None)
        elif (output_format == 'HDF5') | (output_format == 'H5'):
            self.tags_dataframe.to_hdf(tags_fname, key='timeseries', mode='w', format='fixed')
        elif (output_format == 'PARQUET'):
            table = pa.Table.from_pandas(self.tags_dataframe)
            pq.write_table(table, tags_fname)
            del table
            
        self.all_tags_fname = tags_fname
        
        size = utils.get_readable_filesize(os.stat(tags_fname).st_size)
        if self.verbose == 1:
            print(f'File saved: {tags_fname} ({size})')
        
    def get_tags_list(self):
        """
        This functions builds a list of tags present in the tags dataframe. 
        The tags list is determined by taking all unique values in the tag 
        field.
        
        RETURNS
        =======
            tags_list (list)
                Name of all tags found in the tags dataset
                
        EXCEPTIONS
        ==========
            General exceptions is raised if no dataset has been loaded with 
            the build_tags_dataframe() method
        """
        if self.tags_dataframe is None:
            raise Exception('No dataset loaded. Use the build_tags_dataframe() method to load a dataset from disk.')

        if self.tags_list is None:
            if self.input_format == 'stream':
                self.tags_list = self.tags_dataframe[self.raw_data_schema['tag']].unique()
            elif self.input_format == 'tabular':
                self.tags_list = list(self.tags_dataframe.columns)[1:]
            
        return self.tags_list
    
    def get_timeseries(self, tag):
        """
        Loads the time series for a single tag into a dataframe.
        
        PARAMS
        ======
            tag (string)
                Tag name for which we want the time series
                
        RETURNS
        =======
            tag_df (dataframe)
                A dataframe with 2 columns: Timestamp and tag name (the 
                latter, containing the time series values for the desired 
                tag)
                
        EXCEPTIONS
        ==========
            General exceptions is raised:
            - if no dataset has been loaded with the build_tags_dataframe() method
            - if no schema has been defined with define_raw_data_schema() method
        """
        if self.tags_dataframe is None:
            raise Exception('No dataset loaded. Use the build_tags_dataframe() method to load a dataset from disk.')

        if self.raw_data_schema is None:
            raise Exception('Undefined schema, run define_raw_data_schema() before building the tags dataset.')
            
        timestamp = self.raw_data_schema['timestamp']
        if self.input_format == 'stream':
            tag_df = self.tags_dataframe.loc[self.tags_dataframe[self.raw_data_schema['tag']] == tag, :].copy()
        elif self.input_format == 'tabular':
            tag_df = self.tags_dataframe[[timestamp, tag]].copy()
        
        # Transform timestamp to Hogun compatible format:
        tag_df[timestamp] = pd.to_datetime(tag_df[timestamp], format='%Y-%m-%d %H:%M:%S')
        tag_df[timestamp] = tag_df[timestamp].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        
        # We only keep the (deduplicated) timestamp and tag values:
        if self.input_format == 'stream':
            cols_to_delete = [self.raw_data_schema['tag']]
            if (self.raw_data_schema['state'] is not None) and (self.raw_data_schema['state'] != ''):
                cols_to_delete = cols_to_delete + [self.raw_data_schema['state']]                                       
            tag_df.drop(cols_to_delete, axis='columns', inplace=True)
        
        tag_df.columns = ['Timestamp', tag]
        tag_df.drop_duplicates(subset='Timestamp', keep='first', inplace=True)
        return tag_df
    
    def load_file(self, filename):
        if self.output_format == 'CSV':
            df = pd.read_csv(filename)
            
        elif self.output_format == 'HDF5':
            df = pd.read_hdf(filename, key='timeseries', mode='r')
            
        elif self.output_format == 'PARQUET':
            table = pq.read_table(filename)
            df = table.to_pandas()
            del table
            
        return df
    
    def load_timeseries(self, tag):
        """
        Loads a single tag dataframe from disk
        
        PARAMS
        ======
            tag (string)
                We will search for the file in the target_data_root_dir
                folder either at the root or in the component subdirectory.
                
        RETURNS
        =======
            tag_df - A dataframe containing the time series associated 
            to this tag
        """
        extension = self.get_extension(self.output_format)
        
        if self.components_df is not None:
            component = self.components_df.loc[self.components_df['Tag'] == tag, 'Component'].iloc[0]
            tag_fname = os.path.join(self.target_data_root_dir, self.component_prefix + component, tag + extension)
        else:
            tag_fname = os.path.join(self.target_data_root_dir, tag + extension)
            
        tag_df = self.load_file(tag_fname)
        
        if self.output_format != 'PARQUET':
            tag_df.columns = ['Timestamp', 'Value']
            tag_df['Timestamp'] = pd.to_datetime(tag_df['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
            tag_df = tag_df.set_index('Timestamp')
        
        return tag_df
    
    def save_timeseries(self, tag_df, component=None, overwrite=False, output_format='CSV'):
        """
        Saves a single tag dataframe to disk.
        
        PARAMS
        ======
            tag_df (dataframe)
                Dataframe containing the time series for a single tag 
                (timestamp and values stored in tag name, no index)
                
            overwrite (boolean)
                Files will be overwritten if True (default: False)
                
            output_format (string)
                CSV to save to .csv format (default behavior), HDF5 
                (saved as .h5 file) or PARQUET (saved as .parquet file)
        """
        # The tag name is the column name of the second column 
        # of the dataframe (the first one being the timestamp):
        tag = tag_df.columns.tolist()[-1]
        extension = self.get_extension(output_format)
        if component is None:
            tag_fname = os.path.join(self.target_data_root_dir, tag + extension)
            
        else:
            os.makedirs(os.path.join(self.target_data_root_dir, self.component_prefix + component), exist_ok=True)
            tag_fname = os.path.join(self.target_data_root_dir, self.component_prefix + component, tag + extension)
        
        if (not os.path.exists(tag_fname)) | (overwrite == True):
            if output_format == 'CSV':
                tag_df.to_csv(tag_fname, index=None)
            elif (output_format == 'HDF5') | (output_format == 'H5'):
                tag_df.to_hdf(tag_fname, key='timeseries', mode='w', format='fixed')
            elif (output_format == 'PARQUET'):
                table = pa.Table.from_pandas(tag_df, preserve_index=True)
                pq.write_table(table, tag_fname)
                del table
        
    def save_all_timeseries(self, 
                            overwrite=False, 
                            components_df=None, 
                            tag_col='Tag',
                            component_col='Subsystem', 
                            component_prefix='component-',
                            output_format='CSV'
                           ):
        """
        Loops through all the tags found in the main tags dataframe and 
        export each of them in an individual file saved to disk.
        
        PARAMS
        ======
            overwrite (boolean)
                States if we want to overwrite existing file or not, 
                defaults to False (do not overwrite)
            
            components_df (dataframe)
                A dataframe where we can find the tags hierarchy. We need
                at least a "Tag" and a "Component" columns in this dataframe.
                Names of these columns can be configured.
                
            tag_col (string):
                Name of the column where we can find the tag name. Defaults
                to "Tag"
                
            component_col (string)
                Name of the column where we can find the component name.
                Defaults to "Subsystem"
                
            component_prefix (string)
                If a tag hierarchy description dataframe is provided, we will
                save the tags in a folder named after their component. This is
                a string used to prefix the component name when creating the 
                folder.
                
            output_format (string)
                CSV to save to .csv format (default behavior), HDF5 
                (saved as .h5 file) or PARQUET (saved as .parquet file)
                
        EXCEPTIONS
        ==========
            Generic exceptions are raised if:
            - No dataset was loaded
            - If columns tag_col or component_col are not found in components_df
        """
        if self.tags_dataframe is None:
            raise Exception('No dataset loaded. Use the build_tags_dataframe() method to load a dataset from disk.')

        if components_df is not None:
            if tag_col not in components_df:
                raise Exception(f'Column {tag_col} not found in the components_df dataframe.')
            if component_col not in components_df:
                raise Exception(f'Column {component_col} not found in the components_df dataframe.')
                
            self.component_prefix = component_prefix
            self.components_df = components_df[[tag_col, component_col]]
            self.components_df.columns = ['Tag', 'Component']
            
        else:
            self.component_prefix = None
            self.components_df = None
                        
        if self.tags_list is None:
            self.get_tags_list()
            
        # Loops through the tags list and write each of the them to disk:
        for index in tqdm(range(len(self.tags_list)), desc='Saving time series to disk'):
            tag = self.tags_list[index]
            extension = self.get_extension(output_format)
            
            # Get a component name if we provided a tag hierarchy description:
            if components_df is not None:
                component = components_df.loc[components_df[tag_col] == tag, component_col].iloc[0]
                tag_fname = os.path.join(self.target_data_root_dir, component_prefix + component, tag + extension)
                
            else:
                component = None
                tag_fname = os.path.join(self.target_data_root_dir, tag + extension)
                
            # Write the time series associated to the current tag:
            if (not os.path.exists(tag_fname)) | (overwrite == True):
                tag_df = self.get_timeseries(tag)
                
                if self.sampling_rate is not None:
                    tag_df['Timestamp'] = pd.to_datetime(tag_df['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
                    tag_df = tag_df.set_index('Timestamp')
                    tag_df = tag_df.resample(self.sampling_rate).mean()
                    
                    if self.off_timestamps is not None:
                        on_timestamps = tag_df.index.drop(self.off_timestamps)
                        tag_df = tag_df.loc[on_timestamps, :]
                        
                    if output_format != 'PARQUET':
                        tag_df = tag_df.reset_index()

                self.save_timeseries(tag_df, component, overwrite, output_format)
                
        self.timeseries_saved = True
                
    def assign_off_tag(self, tag, threshold, tolerance=None):
        """
        One can use this function to remove any segment of time to be 
        ignored. This can be done by using a guiding signal (one of the
        loaded tag) and a threshold: any portion of time for which this
        guiding signal is below a threshold will be considered an "off"
        time.
        
        PARAMS
        ======
            tag (string)
                Name of the tag to be used as the underlying guide
            
            threshold (float)
                Every timestamps for which the tag value is under
                this threshold we be ignored.
        """
        # Tag we want to use as a guide to identify when 
        # the asset or the process is in an off state:
        self.onoff_tag = tag
        
        # If each individual timeseries was already saved to
        # disk in an optimized format, we load it from there:
        if self.timeseries_saved:
            tag_df = self.load_timeseries(tag)
            
        # Otherwise, we extract it from the master dataframe:
        else:
            tag_df = self.get_timeseries(tag)
        #######################################################################
        ### TODO ### Check if it makes sense to revert to load() from disk from
        #            within the get() method.
        #######################################################################
        
        # Now we can identify the timestamps with values
        # below the threshold passed as argument:
        if self.sampling_rate is not None:
            tag_df_cleaned = tag_df.copy()
            if not isinstance(tag_df_cleaned.index, pd.core.indexes.datetimes.DatetimeIndex):
                tag_df_cleaned['Timestamp'] = pd.to_datetime(tag_df_cleaned['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
                tag_df_cleaned = tag_df_cleaned.set_index('Timestamp')
            tag_df_cleaned = tag_df_cleaned.resample(self.sampling_rate).min()
            tag_df_cleaned = tag_df_cleaned[tag_df_cleaned[tag] > threshold].copy()
            tag_df_cleaned = tag_df_cleaned.resample(self.sampling_rate).mean()
            
        else:
            tag_df_cleaned = tag_df[tag_df[tag] > threshold].copy()
            
        off_timestamps = tag_df_cleaned[tag_df_cleaned[tag].isna()].index
    
        diff = pd.DataFrame(off_timestamps.to_series().diff())
        diff['Delta'] = diff['Timestamp'] == self.sampling_rate
        diff['Next'] = diff['Timestamp'].shift(-1)
        diff['Next Delta'] = diff['Next'] == self.sampling_rate
        diff['Start'] = (diff['Delta'] == False) & (diff['Next Delta'] == True)
        diff['End'] = (diff['Delta'] == True) & (diff['Next Delta'] == False)
        diff = diff[(diff['Start'] == True) | (diff['End'] == True)]
        
        ignored_sequences_df = pd.DataFrame(columns=['Start', 'End'])
        off = None
        for index, row in diff.iterrows():
            if row['Start']:
                if tolerance is None:
                    start = index
                else:
                    start = index - timedelta(minutes=15)
            if row['End']:
                if tolerance is None:
                    end = index
                else:
                    end = index + timedelta(minutes=15)
                
                ignored_sequences_df = ignored_sequences_df.append({
                    'Start': start,
                    'End': end
                }, ignore_index=True)
                current_off = pd.date_range(start=start, end=end, freq=self.sampling_rate)
                if off is None:
                    off = current_off
                else:
                    off = off.union(current_off)

        # Cleanup
        del tag_df, tag_df_cleaned, diff
        self.ignored_sequences_df = ignored_sequences_df
        self.off_timestamps = off
                
    def set_label_ranges(self, labels_df):
        """
        Associate a label of anomaly ranges to this dataset.
        
        PARAMS
        ======
            labels_df (dataframe)
                A dataframe with an anomaly range per row, with a start 
                and end columns.
        """
        self.label_ranges = labels_df
        
    def __del__(self):
        """
        Delete 
        """
        del self.raw_data_schema
        del self.tags_dataframe
        del self.tags_list