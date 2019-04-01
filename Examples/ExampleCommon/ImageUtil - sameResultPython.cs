using System.IO;
using TensorFlow;

namespace ExampleCommon
{
	public static class ImageUtil
	{
		// Convert the image in filename to a Tensor suitable as input to the Inception model.
		public static TFTensor CreateTensorFromImageFile (string file, TFDataType destinationDataType = TFDataType.Float)
		{
			var contents = File.ReadAllBytes (file);

			// DecodeJpeg uses a scalar String-valued tensor as input.
			var tensor = TFTensor.CreateString (contents);

			TFOutput input, output;

			// Construct a graph to normalize the image
			using (var graph = ConstructGraphToNormalizeImage (out input, out output, destinationDataType)){
				// Execute that graph to normalize this one image
				using (var session = new TFSession (graph)) {
					var normalized = session.Run (
						inputs: new [] { input },
						inputValues: new [] { tensor },
						outputs: new [] { output });
					
					return normalized [0];
				}
			}
		}

		// The inception model takes as input the image described by a Tensor in a very
		// specific normalized format (a particular image size, shape of the input tensor,
		// normalized pixel values etc.).
		//
		// This function constructs a graph of TensorFlow operations which takes as
		// input a JPEG-encoded string and returns a tensor suitable as input to the
		// inception model.
		private static TFGraph ConstructGraphToNormalizeImage (out TFOutput input, out TFOutput output, TFDataType destinationDataType = TFDataType.Float)
		{
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained after with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.


            var graph = new TFGraph();
            input = graph.Placeholder(TFDataType.String);

            output = graph.Cast(
                        graph.ExpandDims(
                                input: graph.Cast(graph.DecodeJpeg(contents: input, channels: 3), DstT: TFDataType.Float),
                                dim: graph.Const(0, "make_batch")
                       )
                       , destinationDataType
                    );

            return graph;

            //const int W = 227;
            //const int H = 227;
            //const float Scale = 1;

            //// Depending on your CustomVision.ai Domain - set appropriate Mean Values (RGB)
            //// https://github.com/Azure-Samples/cognitive-services-android-customvision-sample for RGB values (in BGR order)
            //var bgrValues = new TFTensor(new float[] { 104.0f, 117.0f, 123.0f }); // General (Compact) & Landmark (Compact)
            ////var bgrValues = new TFTensor(0f); // Retail (Compact)

            //var graph = new TFGraph();
            //input = graph.Placeholder(TFDataType.String);

            //var caster = graph.Cast(graph.DecodeJpeg(contents: input, channels: 3), DstT: TFDataType.Float);
            //var dims_expander = graph.ExpandDims(caster, graph.Const(0, "batch"));
            //var resized = graph.ResizeBilinear(dims_expander, graph.Const(new int[] { H, W }, "size"));
            //var resized_mean = graph.Sub(resized, graph.Const(bgrValues, "mean"));
            //var normalised = graph.Div(resized_mean, graph.Const(Scale));
            //output = normalised;
            //return graph;
        }
	}
}
