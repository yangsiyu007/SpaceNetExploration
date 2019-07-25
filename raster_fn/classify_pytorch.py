import building_eval_lib as eval
import numpy as np

class ClassifyPyTorch():

    def __init__(self):
        self.name = "Classify PyTorch Function"
        self.description = ("This function classifies each pixel as one of building, "
                            "boundary of building or background.")

        self.model = None
        self.padding = 0
        self.hard = None
        self.batch = 8

    def getParameterInfo(self):
        return [
            {
                'name': 'input',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "A 3-band raster from a RGB satellite or aerial image"
            },
            {
                'name': 'model',
                'dataType': 'string',
                'value': None,
                'required': True,
                'displayName': "Trained PyTorch model",
                'description': ""
            },
            {
                'name': 'batch',
                'dataType': 'numeric',
                'value': 8,
                'required': True,
                'displayName': "Mini batch size",
                'description': "How many blocks of 256x256 squares are evaluated at once"
            },
            {
                'name': 'hard',
                'dataType': 'string',
                'value': None,
                'required': True,
                'displayName': "Output Type",
                'domain': ('Softmax', 'Hardmax'),
                'description': "Whether the output should be probabilities (softmax) or classifications (hardmax)"
            }
        ]

    def getConfiguration(self, **scalars):
        return {
            'inheritProperties': 4 | 2,
            'invalidateProperties': 2 | 4 | 8,
            'padding': self.padding
            # 'inputMask': False                         # Need input raster mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):
        self.model = eval.load_model(kwargs.get('model', None))
        _, _, self.padding = eval.get_train_info(self.model)
        self.batch = kwargs.get('batch', 128)
        self.hard = kwargs.get('hard')

        kwargs['output_info']['bandCount'] = 3
        kwargs['output_info']['statistics'] = ()  # we know nothing about the stats of the outgoing raster.
        kwargs['output_info']['histogram'] = ()  # we know nothing about the histogram of the outgoing raster.
        kwargs['output_info']['pixelType'] = 'u1'
        kwargs['output_info']['resampling'] = False

        return kwargs

    def updatePixels(self, tlc, size, props, **pixelBlocks):
        tile_data = np.array(pixelBlocks['input_pixels'], dtype='f4', copy=False)  # do not divide by 256 - the model was trained with pixels in 0 - 255
        tile_data_3_bands = tile_data[0:3, :, :]
        b, w, h = tile_data_3_bands.shape
        num_classes = 3
        output_data = np.zeros((num_classes, w, h), dtype=np.float32)  # we want the probability of each class

        eval.classify_tile(self.model, tile_data_3_bands, output_data, self.batch)

        # used this line when there is padding
        pd = int(self.padding)
        if pd > 0:
            output_data = output_data[:, pd:-pd, pd:-pd]

        img = eval.render(output_data, self.hard == 'Hardmax')
        img = np.round(img * 255, 0).astype(np.uint8)
        img = img.astype(props['pixelType'], copy=True)
        pixelBlocks['output_pixels'] = img

        return pixelBlocks