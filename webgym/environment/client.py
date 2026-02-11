import requests
from PIL import Image
import io
import ipywidgets
import json
import time
from . import actions
from IPython.display import display
import urllib3

# Suppress SSL warnings when using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class InstanceClient:
    def __init__(self, host = 'localhost', port = 5000):
        self.host = host
        self.port = port
        self.output = None

    def _get_base_url(self):
        """Get the base URL with appropriate protocol (HTTPS for port 443, HTTP otherwise)"""
        protocol = 'https' if self.port == 443 else 'http'
        return f'{protocol}://{self.host}:{self.port}'

    def screenshot(self):
        data = requests.get(f'{self._get_base_url()}/screenshot', timeout=None)
        image_data = io.BytesIO(data.content)
        return Image.open(image_data)

    def execute(self, command):
        return requests.post(f'{self._get_base_url()}/execute', json = command)
    
    def do_and_show(self, command, waitTime):
        response = self.execute(command)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        time.sleep(waitTime)
        self._display(response.json()['output'])
    
    def _display(self, data = None):
        with self.output:
            self.output.clear_output(wait = True)
            display(self.screenshot())
            print(data)

    def position(self):
        return json.loads(self.execute(actions.position()).json()['output'])
    
    def screensize(self):
        return json.loads(self.execute(actions.screensize()).json()['output'])    

    def ui(self):
        self.output = ipywidgets.Output()
        self._display()

        screensize = self.screensize()
        position = self.position()

        waitTime = ipywidgets.FloatLogSlider(base=2, value = 1, min = -3, max = 5, step = 1, description = 'wait (s)')

        x = ipywidgets.IntSlider(min = 0, max = screensize.get('width', 1), value = position.get('x', 0), description = 'X')
        y = ipywidgets.IntSlider(min = 0, max = screensize.get('height', 1), value = position.get('y', 0), description = 'Y')

        click = ipywidgets.Button(description = 'Click')
        rightClick = ipywidgets.Button(description = 'Right Click')
        doubleClick = ipywidgets.Button(description = 'Double Click')

        x.observe(lambda v: self.do_and_show(actions.moveTo(x.value, y.value), waitTime.value), names='value')
        y.observe(lambda v: self.do_and_show(actions.moveTo(x.value, y.value), waitTime.value), names='value')

        click.on_click(lambda v: self.do_and_show(actions.click(), waitTime.value))
        rightClick.on_click(lambda v: self.do_and_show(actions.rightClick(), waitTime.value))
        doubleClick.on_click(lambda v: self.do_and_show(actions.doubleClick(), waitTime.value))

        cmd = ipywidgets.Textarea(description = 'Commands', layout=ipywidgets.Layout(width='50%'))
        shell = ipywidgets.Checkbox(description = 'Shell', value = False)
        python = ipywidgets.Checkbox(description = 'Python', value = True)
        submit = ipywidgets.Button(description = 'Submit')
        submit.on_click(lambda v: self.do_and_show({
            'command': actions._pyscript(cmd.value.split('\n')) if python.value else '&&'.join(cmd.value.split('\n')),
            'shell': shell.value
        }, waitTime.value))
        display(ipywidgets.VBox([
            waitTime,
            ipywidgets.HBox([x, y]),
            ipywidgets.HBox([click, rightClick, doubleClick]),
            ipywidgets.HBox([cmd, ipywidgets.VBox([shell, python]), submit]),
            self.output]))
        

class NodeClient:
    def __init__(self, host = 'localhost', port = 8000):
        self.host = host
        self.port = port
        self.output = None
        self.instance_id = None

    def _get_base_url(self):
        """Get the base URL with appropriate protocol (HTTPS for port 443, HTTP otherwise)"""
        protocol = 'https' if self.port == 443 else 'http'
        return f'{protocol}://{self.host}:{self.port}'

    def get_instance(self):
        data = requests.post(f'{self._get_base_url()}/get').json()
        return data.get('instance_id', None)

    def get_instances_info(self):
        data = requests.get(f'{self._get_base_url()}/info')
        return data.json()

    def reset_instance(self, instance_id):
        data = requests.post(f'{self._get_base_url()}/reset', params = {'instance_id': instance_id})
        if data.status_code != 200:
            raise Exception(f"Error: {data.status_code} - {data.text}")
        return data.json()

    def screenshot(self, instance_id):
        data = requests.get(f'{self._get_base_url()}/screenshot', params = {'instance_id': instance_id}, timeout=None)
        image_data = io.BytesIO(data.content)
        return Image.open(image_data)

    def execute(self, instance_id, command):
        return requests.post(
            f'{self._get_base_url()}/execute',
            params={'instance_id': instance_id},
            json = command)
    
    def do_and_show(self, instance_id, command, waitTime):
        response = self.execute(instance_id, command)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}") 
        time.sleep(waitTime)
        self._display(instance_id, response.json())
    
    def _display(self, instance_id, data = None):
        with self.output:
            self.output.clear_output(wait = True)
            display(self.screenshot(instance_id))
            print(data)

    def position(self, instance_id):
        data = requests.get(f'{self._get_base_url()}/metadata', params = {'instance_id': instance_id}).json()
        return {'x': data.get('x', 0), 'y': data.get('y', 0)}

    def screensize(self, instance_id):
        data = requests.get(f'{self._get_base_url()}/metadata', params = {'instance_id': instance_id}).json()
        return {'width': data['width'], 'height': data['height']} 
    
    def ui(self):
        self.output = ipywidgets.Output()

        get_button = ipywidgets.Button(description = 'Get Instance')
        release_button = ipywidgets.Button(description = 'Release Instance')
        instances = ipywidgets.RadioButtons(description = 'Instances', options = self.get_instances_info()['in_use'], layout=ipywidgets.Layout(width='50%'))
        self.instance_id = instances.value
        def on_add_click(_):
            instance_id = self.get_instance()
            if instance_id:
                instances.options = list(instances.options) + [instance_id]
            else:
                print('No instance available')

        def on_release_click(_):
            self.reset_instance(instances.value)
            options = list(instances.options)
            options.remove(instances.value)
            instances.options = options       

        get_button.on_click(on_add_click)
        release_button.on_click(on_release_click)        

        waitTime = ipywidgets.FloatLogSlider(base=2, value = 1, min = -3, max = 5, step = 1, description = 'wait (s)')

        screensize = {}
        position = {}
        if self.instance_id:
            screensize = self.screensize(self.instance_id)
            position = self.position(self.instance_id)
            self._display(self.instance_id)

        x = ipywidgets.IntSlider(min = 0, max = screensize.get('width', 1), value = position.get('x', 0), description = 'X')
        y = ipywidgets.IntSlider(min = 0, max = screensize.get('height', 1), value = position.get('y', 0), description = 'Y')

        click = ipywidgets.Button(description = 'Click')
        rightClick = ipywidgets.Button(description = 'Right Click')
        doubleClick = ipywidgets.Button(description = 'Double Click')

        x.observe(lambda v: self.do_and_show(self.instance_id, actions.moveTo(x.value, y.value), waitTime.value), names='value')
        y.observe(lambda v: self.do_and_show(self.instance_id, actions.moveTo(x.value, y.value), waitTime.value), names='value')

        click.on_click(lambda v: self.do_and_show(self.instance_id, actions.click(), waitTime.value))
        rightClick.on_click(lambda v: self.do_and_show(self.instance_id, actions.rightClick(), waitTime.value))
        doubleClick.on_click(lambda v: self.do_and_show(self.instance_id, actions.doubleClick(), waitTime.value))

        cmd = ipywidgets.Textarea(description = 'Commands', layout=ipywidgets.Layout(width='50%'))
        shell = ipywidgets.Checkbox(description = 'Shell', value = False)
        python = ipywidgets.Checkbox(description = 'Python', value = False)
        is_json = ipywidgets.Checkbox(description = 'Json', value = True)
        submit = ipywidgets.Button(description = 'Submit')

        def _prepare_cmd():
            if is_json.value:
                return json.loads(cmd.value)
            else:
                return {
                    'command': actions._pyscript(cmd.value.split('\n')) if python.value else '&&'.join(cmd.value.split('\n')),
                    'shell': shell.value
                }
        submit.on_click(lambda v: self.do_and_show(self.instance_id, _prepare_cmd(), waitTime.value))

        def on_instance_change(_):
            self.instance_id = instances.value
            screensize = self.screensize(self.instance_id)
            x.max = screensize.get('width', 1)
            y.max = screensize.get('height', 1)
            position = self.position(self.instance_id)
            x.value = position.get('x', 0)
            y.value = position.get('y', 0)
            self._display(self.instance_id)

        instances.observe(on_instance_change, names='value')

        display(ipywidgets.VBox([
            ipywidgets.HBox([instances, ipywidgets.VBox([get_button, release_button])]),
            waitTime,
            ipywidgets.HBox([x, y]),
            ipywidgets.HBox([click, rightClick, doubleClick]),
            ipywidgets.HBox([cmd, ipywidgets.VBox([shell, python, is_json]), submit]),
            self.output]))
        

class MasterClient:
    def __init__(self, host = 'localhost', port = 7000, api_key = None):
        self.host = host
        self.port = port
        self.output = None
        self.instance = None
        self.api_key = api_key

    def _get_base_url(self):
        """Get the base URL with appropriate protocol (HTTPS for port 443, HTTP otherwise)"""
        protocol = 'https' if self.port == 443 else 'http'
        return f'{protocol}://{self.host}:{self.port}'

    def probe(self, instance: dict):
        return requests.get(f'{self._get_base_url()}/probe', params=instance, headers={"x-api-key": self.api_key}, verify=False).json()

    def get_instance(self, lifetime_mins = 120):
        url = f'{self._get_base_url()}/get'
        response = requests.post(url, params={'lifetime_mins': lifetime_mins}, headers={"x-api-key": self.api_key}, verify=False)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        return response.json()

    def get_info(self):
        url = f'{self._get_base_url()}/info'
        response = requests.get(url, headers={"x-api-key": self.api_key}, verify=False)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        return response.json()
    
    def get_options(self):
        info = self.get_info()['nodes']
        print(type(info))
        return [json.dumps({'instance_id': instance, 'node': node['hash']}) for node in info for instance in node['instances']]

    def reset_instance(self, instance: dict):
        data = requests.post(f'{self._get_base_url()}/reset', params = instance, headers={"x-api-key": self.api_key}, verify=False)
        if data.status_code != 200:
            raise Exception(f"Error: {data.status_code} - {data.text}")
        return data.json()

    def screenshot(self, instance: dict, interaction_mode: str = "set_of_marks", stream: bool = False):
        if stream:
            # Use streaming to avoid buffer exhaustion
            response = requests.get(f'{self._get_base_url()}/screenshot', 
                                  params=dict(instance, **{'interaction_mode': interaction_mode}), 
                                  headers={"x-api-key": self.api_key}, 
                                  verify=False, timeout=None, stream=True)
            
            if response.status_code != 200:
                raise Exception(f"Screenshot request failed: {response.status_code} - {response.text}")
            
            screenshot_data = b''
            for chunk in response.iter_content(chunk_size=65536):  # 64KB chunks
                if chunk:
                    screenshot_data += chunk
            
            # Convert bytes to PIL Image
            image_data = io.BytesIO(screenshot_data)
            return Image.open(image_data)
        else:
            # Original implementation for backward compatibility
            data = requests.get(f'{self._get_base_url()}/screenshot', params = dict(instance, **{'interaction_mode': interaction_mode}), headers={"x-api-key": self.api_key}, verify=False, timeout=None)
            if data.status_code != 200:
                raise Exception(f"Screenshot request failed: {data.status_code} - {data.text}")
            image_data = io.BytesIO(data.content)
            return Image.open(image_data)

    def metadata(self, instance: dict):
        """Get instance viewport metadata (width, height)"""
        response = requests.get(f'{self._get_base_url()}/metadata', params=instance, headers={"x-api-key": self.api_key}, verify=False)
        if response.status_code != 200:
            raise Exception(f"Metadata request failed: {response.status_code} - {response.text}")
        return response.json()

    def execute(self, instance: dict, command):
        return requests.post(f'{self._get_base_url()}/execute', json = dict(command, **instance), headers={"x-api-key": self.api_key}, verify=False)

    def do_and_show(self, instance, command, waitTime):
        response = self.execute(instance, command)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        time.sleep(waitTime)
        self._display(instance, response.json())
    
    def _display(self, instance, data = None):
        with self.output:
            self.output.clear_output(wait = True)
            display(self.screenshot(instance))
            print(data)

    def position(self, instance):
        data = requests.get(f'{self._get_base_url()}/metadata', params = instance, headers={"x-api-key": self.api_key}, verify=False).json()
        return {'x': data.get('x', 0), 'y': data.get('y', 0)}

    def screensize(self, instance):
        data = requests.get(f'{self._get_base_url()}/metadata', params = instance, headers={"x-api-key": self.api_key}, verify=False).json()
        return {'width': data['width'], 'height': data['height']}  

    def ui(self):
        self.output = ipywidgets.Output()

        lifetime = ipywidgets.IntSlider(min = 0, max = 60, value = 60, description = 'Lifetime (min)')
        get_button = ipywidgets.Button(description = 'Get Instance')
        release_button = ipywidgets.Button(description = 'Release Instance')
        instances = ipywidgets.RadioButtons(description = 'Instances', options = self.get_options(), layout=ipywidgets.Layout(width='50%'))
        self.instance = json.loads(instances.value) if instances.value else {}
        def on_add_click(_):
            new_instance = json.dumps(self.get_instance(lifetime_mins = lifetime.value))
            instances.options = list(instances.options) + [new_instance]

        def on_release_click(_):
            self.reset_instance(json.loads(instances.value))
            options = list(instances.options)
            options.remove(instances.value)
            instances.options = options        

        get_button.on_click(on_add_click)
        release_button.on_click(on_release_click)        

        waitTime = ipywidgets.FloatLogSlider(base=2, value = 1, min = -3, max = 5, step = 1, description = 'wait (s)')

        screensize = {}
        position = {}
        if self.instance:
            screensize = self.screensize(self.instance)
            position = self.position(self.instance)
            self._display(self.instance)

        x = ipywidgets.IntSlider(min = 0, max = screensize.get('width', 1), value = position.get('x', 0), description = 'X')
        y = ipywidgets.IntSlider(min = 0, max = screensize.get('height', 1), value = position.get('y', 0), description = 'Y')

        click = ipywidgets.Button(description = 'Click')
        rightClick = ipywidgets.Button(description = 'Right Click')
        doubleClick = ipywidgets.Button(description = 'Double Click')

        x.observe(lambda v: self.do_and_show(self.instance, actions.moveTo(x.value, y.value), waitTime.value), names='value')
        y.observe(lambda v: self.do_and_show(self.instance, actions.moveTo(x.value, y.value), waitTime.value), names='value')

        click.on_click(lambda v: self.do_and_show(self.instance, actions.click(), waitTime.value))
        rightClick.on_click(lambda v: self.do_and_show(self.instance, actions.rightClick(), waitTime.value))
        doubleClick.on_click(lambda v: self.do_and_show(self.instance, actions.doubleClick(), waitTime.value))

        cmd = ipywidgets.Textarea(description = 'Commands', layout=ipywidgets.Layout(width='50%'))
        shell = ipywidgets.Checkbox(description = 'Shell', value = False)
        python = ipywidgets.Checkbox(description = 'Python', value = False)
        is_json = ipywidgets.Checkbox(description = 'Json', value = True)
        submit = ipywidgets.Button(description = 'Submit')

        def _prepare_cmd():
            if is_json.value:
                return json.loads(cmd.value)
            else:
                return {
                    'command': actions._pyscript(cmd.value.split('\n')) if python.value else '&&'.join(cmd.value.split('\n')),
                    'shell': shell.value
                }
        submit.on_click(lambda v: self.do_and_show(self.instance, _prepare_cmd(), waitTime.value))
        refresh = ipywidgets.Button(description = 'Refresh')
        def _on_refresh_click(_):
            instances.options = self.get_options()
        refresh.observe(_on_refresh_click, names='value')

        def on_instance_change(_):
            self.instance = json.loads(instances.value)
            screensize = self.screensize(self.instance)
            x.max = screensize.get('width', 1)
            y.max = screensize.get('height', 1)
            position = self.position(self.instance)
            x.value = position.get('x', 0)
            y.value = position.get('y', 0)
            self._display(self.instance)

        instances.observe(on_instance_change, names='value')

        display(ipywidgets.VBox([
            ipywidgets.HBox([instances, ipywidgets.VBox([lifetime, get_button, release_button, refresh])]),
            waitTime,
            ipywidgets.HBox([x, y]),
            ipywidgets.HBox([click, rightClick, doubleClick]),
            ipywidgets.HBox([cmd, ipywidgets.VBox([shell, python, is_json]), submit]),
            self.output]))