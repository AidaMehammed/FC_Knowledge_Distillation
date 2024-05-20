
from FeatureCloud.app.engine.app import AppState, app_state, Role
from typing import TypedDict
from torch.utils.data import Dataset, DataLoader
import utils as pf
import torch
import bios
import importlib.util



INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'



class KDType(TypedDict):
    teacher_model: torch.nn.Module
    student_model: torch.nn.Module
    train_loader: DataLoader
    epochs: int
    learning_rate: float
    temperature: float
    alpha: float



class KDAppState(AppState):



    def configure_kd(self, teacher_model: torch.nn.Module = None, student_model: torch.nn.Module = None, train_loader: DataLoader = None,
                     epochs: int = 0, learning_rate: float = 0.001, temperature: float = 1.0, alpha: float = 0.5):
        '''
        Configures the knowledge distillation settings for your model.

        Parameters
        ----------
        teacher_model : torch.nn.Module, optional
            Teacher model for knowledge distillation. Default is None.
        student_model : torch.nn.Module, optional
            Student model for knowledge distillation. Default is None.
        train_loader : DataLoader, optional
            DataLoader for training data. Default is None.
        epochs : int, optional
            Number of training epochs for knowledge distillation. Default is 0.
        learning_rate : float, optional
            Learning rate for knowledge distillation. Default is 0.001.
        temperature : float, optional
            Temperature parameter for knowledge distillation. Default is 1.0.
        alpha : float, optional
            Alpha parameter for knowledge distillation. Default is 0.5.
            Should be in the range [0, 1]
        '''

        if self.load('default_kd') is None:
            self.store('default_kd', KDType())

        default_kd = self.load('default_kd')

        updated_kd = default_kd.copy()

        updated_kd['teacher_model'] = teacher_model
        updated_kd['student_model'] = student_model
        updated_kd['train_loader'] = train_loader
        updated_kd['epochs'] = epochs
        updated_kd['learning_rate'] = learning_rate
        updated_kd['temperature'] = temperature
        updated_kd['alpha'] = alpha

        self.store('default_kd', updated_kd)





    def send_data_to_coordinator(self, data, use_kd = True, **kwargs):
        '''
            Sends data to the coordinator, including knowledge distillation if enabled.

            Parameters
            ----------
            data : list
                List of data to be sent to the coordinator.
            use_kd : bool, optional
                Flag to indicate whether to use nowledge distillation. Default is True.

            Returns
            -------
            data : list
                List of data sent to the coordinator.
            '''

        if use_kd :
            default_kd = self.load('default_kd')
            model = data
            epochs = default_kd['epochs']
            train_loader = default_kd['train_loader']
            learning_rate = default_kd['learning_rate']
            temperature = default_kd['temperature']
            alpha = default_kd['alpha']
            teacher_model = default_kd['teacher_model']

            self.store('default_kd', default_kd)

            self.log('Start Knowledge Distillation...')
            self.log(f'Size of model before KD: {pf.print_size_of_model(teacher_model)} MB')
            # here kd training
            model = pf.train_kd(student_model=model, teacher_model=teacher_model, train_loader=train_loader, T= temperature, alpha=alpha, epochs=epochs, lr=learning_rate)
            data = pf.get_weights(model)

            super().send_data_to_coordinator(data, **kwargs)
        else:
            super().send_data_to_coordinator(data, **kwargs)
        return data



@app_state('initial')
class InitialState(KDAppState):

    def register(self):
        # Register transition for local update
        self.register_transition('local_update',label="Broadcast initial weights")


    def run(self) :

        # Reading configuration file
        self.log('Reading configuration file ...')

        # Loading configuration from file
        config = bios.read(f'{INPUT_DIR}/config.yml')

        max_iterations = config['max_iter']
        self.store('iteration', 0)
        self.store('max_iterations', max_iterations)
        self.store('learning_rate', config['learning_rate'])
        self.store('epochs', config['epochs'])
        self.store('batch_size', config['batch_size'])
        self.store('temperature', config['temperature'])
        self.store('alpha', config['alpha'])

        train_dataset_path = f"{INPUT_DIR}/{config['train_dataset']}"
        test_dataset_path = f"{INPUT_DIR}/{config['test_dataset']}"
        train_dataset = torch.load(train_dataset_path)
        test_dataset = torch.load(test_dataset_path)
        self.store('train_dataset', train_dataset)
        self.store('test_dataset', test_dataset)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.load('batch_size'), shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.load('batch_size'), shuffle=False)
        self.store('train_loader', train_loader)
        self.store('test_loader', test_loader)


        self.log('Done reading configuration.')

        # Loading and preparing initial model
        self.log('Preparing models ...')
        teacher_model_path = f"{INPUT_DIR}/{config['teacher_model']}"

        # Loading model from file
        spec = importlib.util.spec_from_file_location("model_module", teacher_model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model_class_name = config.get('model_class', 'Model')
        teacher_model = getattr(model_module, model_class_name)()

        self.store('teacher_model', teacher_model)

        student_model_path = f"{INPUT_DIR}/{config['student_model']}"

        # Loading model from file
        spec = importlib.util.spec_from_file_location("model_module", student_model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model_class_name = config.get('model_class', 'Model')
        model = getattr(model_module, model_class_name)()

        self.store('model', model)





        self.log('Transition to local state ...')

        if self.is_coordinator:
            # Broadcasting initial weights to participants
            self.broadcast_data([pf.get_weights(model),False], send_to_self=False)
            self.store('received_data', pf.get_weights(model))

        return 'local_update'





@app_state('local_update', Role.BOTH)
class LocalUpdate(KDAppState):
    def register(self):
        # Registering transitions for local update
        self.register_transition('aggregation', Role.COORDINATOR, label="Gather local models")
        self.register_transition('local_update', Role.PARTICIPANT, label="Wait for global model")
        self.register_transition('terminal',label="Terminate process")


    def run(self):
        # Running local update process

        iteration = self.load('iteration')
        self.log(f'ITERATION  {iteration}')
        model = self.load('model')
        stop_flag = False
        if self.is_coordinator:
            received_data = self.load('received_data')
        else:
            received_data, stop_flag = self.await_data(unwrap=True)

        self.log(len(received_data))

        if stop_flag:
            self.log('Stopping')
            return 'terminal'

        # Receive global model from coordinator
        self.log('Receive model from coordinator')

        pf.set_weights(model,received_data)

        # Receive dataframe
        train_loader = self.load('train_loader')
        epochs = self.load('epochs')
        learning_rate = self.load('learning_rate')
        test_loader = self.load('test_loader')
        teacher_model = self.load('teacher_model')
        temperature = self.load('temperature')
        alpha = self.load('alpha')




        self.configure_kd(epochs=epochs, learning_rate=learning_rate, student_model =model, teacher_model= teacher_model, train_loader=train_loader, temperature=temperature, alpha=alpha )

        self.send_data_to_coordinator(model, use_kd=True, use_smpc=False, use_dp=False)

        # Test quantized model
        pf.test(model, test_loader)

        iteration += 1
        self.store('iteration', iteration)

        if stop_flag:
            return 'terminal'

        if self.is_coordinator:
            return 'aggregation'

        else:
            return 'local_update'

@app_state('aggregation', Role.COORDINATOR)
class AggregateState(KDAppState):

    def register(self):
        # Registering transitions for aggregation state
        self.register_transition('local_update', Role.COORDINATOR, label="Broadcast global model")
        self.register_transition('terminal', Role.COORDINATOR, label="Terminate process")

    def run(self) :
        # Running aggregation process
        self.log(f'Aggregating Data ...')
        # Gathering and averaging data
        data = self.gather_data(is_json=False, use_smpc=False, use_dp=False, memo=None)
        self.log(f'Averaging Data ...')

        global_averaged_weights = pf.average_weights(data)

        stop_flag = False
        if self.load('iteration') >= self.load('max_iterations'):
            stop_flag = True

        # Set averaged_weights as new global model
        self.store('received_data', global_averaged_weights)
        new_model= self.load('model')
        pf.set_weights(new_model, global_averaged_weights)

        # Broadcasting global model
        self.log('Broadcasting global model ...')
        self.broadcast_data([global_averaged_weights, stop_flag], send_to_self=False)


        if stop_flag:
            return 'terminal'

        return 'local_update'

