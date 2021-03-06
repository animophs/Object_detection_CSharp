﻿using System;
using System.Collections.Generic;
using TensorFlow;
using System.Text;
using Xunit;
using Learn.Mnist;

namespace TensorFlowSharp.Tests.CSharp
{
	public class TensorTests
	{
		private static IEnumerable<object []> jaggedData ()
		{
			yield return new object [] {
				new double [][] { new [] { 1.0, 2.0 }, new [] { 3.0, 4.0 } },
				new double [,] { { 1.0, 2.0}, { 3.0, 4.0 } },
				true
			};

			yield return new object [] {
				new double [][] { new [] { 1.0, 2.0 }, new [] { 1.0, 4.0 } },
				new double [,] { { 1.0, 2.0}, { 3.0, 4.0 } },
				false
			};

			yield return new object [] {
				new double [][][] { new [] { new [] { 1.0 }, new[] { 2.0 } }, new [] { new [] { 3.0 }, new [] { 4.0 } } },
				new double [,,] { { { 1.0 }, { 2.0 } }, { { 3.0 }, { 4.0 } } },
				true
			};

			yield return new object [] {
				new double [][][] { new [] { new [] { 1.0 }, new[] { 2.0 } }, new [] { new [] { 1.0 }, new [] { 4.0 } } },
				new double [,,] { { { 1.0 }, { 2.0 } }, { { 3.0 }, { 4.0 } } },
				false
			};
		}


		[Theory]
		[MemberData (nameof (jaggedData))]
		public void Should_MultidimensionalAndJaggedBeEqual (Array jagged, Array multidimensional, bool expected)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var tjagged = graph.Const (new TFTensor (jagged));
				var tmultidimensional = graph.Const (new TFTensor (multidimensional));

				TFOutput y = graph.Equal (tjagged, tmultidimensional);
				TFOutput r;
				if (multidimensional.Rank == 2)
					r = graph.All (y, graph.Const (new [] { 0, 1 }));
				else if (multidimensional.Rank == 3)
					r = graph.All (y, graph.Const (new [] { 0, 1, 2 }));
				else
					throw new System.Exception ("If you want to test Ranks > 3 please handle this extra case manually.");

				TFTensor [] result = session.Run (new TFOutput [] { }, new TFTensor [] { }, new [] { r });

				bool actual = (bool)result [0].GetValue ();
				Assert.Equal (expected, actual);
			}
		}

        [Fact]
        public void StringTestWithMultiDimStringTensorAsInputOutput()
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var W = graph.Placeholder(TFDataType.String, new TFShape(-1, 2));
                var identityW  = graph.Identity(W);

                var dataW = new string[,] { { "This is fine.", "That's ok." }, { "This is fine.", "That's ok." } };
                var bytes = new byte[2 * 2][];
                bytes[0] = Encoding.UTF8.GetBytes(dataW[0, 0]);
                bytes[1] = Encoding.UTF8.GetBytes(dataW[0, 1]);
                bytes[2] = Encoding.UTF8.GetBytes(dataW[1, 0]);
                bytes[3] = Encoding.UTF8.GetBytes(dataW[1, 1]);
                var tensorW = TFTensor.CreateString(bytes, new TFShape(2,2));

                var outputTensor = session.Run(new TFOutput[] { W }, new TFTensor[] { tensorW }, new[] { identityW });
                
                var outputW = TFTensor.DecodeMultiDimensionString(outputTensor[0]);
                Assert.Equal(dataW[0, 0], Encoding.UTF8.GetString(outputW[0]));
                Assert.Equal(dataW[0, 1], Encoding.UTF8.GetString(outputW[1]));
                Assert.Equal(dataW[1, 0], Encoding.UTF8.GetString(outputW[2]));
                Assert.Equal(dataW[1, 1], Encoding.UTF8.GetString(outputW[3]));
            }
        }

        [Fact]
        public void StringTestWithMultiDimStringTensorAsInputAndScalarStringAsOutput()
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var X = graph.Placeholder(TFDataType.String, new TFShape(-1));
                var delimiter = graph.Const(TFTensor.CreateString(Encoding.UTF8.GetBytes("/")));
                var indices = graph.Const(0);
                var Y = graph.ReduceJoin(graph.StringSplit(X, delimiter).values, indices, separator: " ");

                var dataX = new string[] { "Thank/you/very/much!.", "I/am/grateful/to/you.", "So/nice/of/you." };
                var bytes = new byte[dataX.Length][];
                bytes[0] = Encoding.UTF8.GetBytes(dataX[0]);
                bytes[1] = Encoding.UTF8.GetBytes(dataX[1]);
                bytes[2] = Encoding.UTF8.GetBytes(dataX[2]);
                var tensorX = TFTensor.CreateString(bytes, new TFShape(3));

                var outputTensors = session.Run(new TFOutput[] { X }, new TFTensor[] { tensorX }, new[] { Y });

                var outputY = Encoding.UTF8.GetString(TFTensor.DecodeString(outputTensors[0]));
                Assert.Equal(string.Join(" ", dataX).Replace("/", " "), outputY);
            }
        }

        [Fact(Skip = "Disabled because it requires GPUs and need to set numGPUs to available GPUs on system." +
            " It has been tested on GPU machine with 4 GPUs and it passed there.")]
        public void DevicePlacementTest()
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var X = graph.Placeholder(TFDataType.Float, new TFShape(-1, 784));
                var Y = graph.Placeholder(TFDataType.Float, new TFShape(-1, 10));

                int numGPUs = 4;
                var Xs = graph.Split(graph.Const(0), X, numGPUs);
                var Ys = graph.Split(graph.Const(0), Y, numGPUs);
                var products = new TFOutput[numGPUs];
                for (int i = 0; i < numGPUs; i++)
                {
                    using (var device = graph.WithDevice("/device:GPU:" + i))
                    {
                        var W = graph.Constant(0.1f, new TFShape(784, 500), TFDataType.Float);
                        var b = graph.Constant(0.1f, new TFShape(500), TFDataType.Float);
                        products[i] = graph.Add(graph.MatMul(Xs[i], W), b);
                    }
                }
                var stacked = graph.Concat(graph.Const(0),products);
                Mnist mnist = new Mnist();
                mnist.ReadDataSets("/tmp");
                int batchSize = 1000;
                for (int i = 0; i < 100; i++)
                {
                    var reader = mnist.GetTrainReader();
                    (var trainX, var trainY) = reader.NextBatch(batchSize);
                    var outputTensors = session.Run(new TFOutput[] { X }, new TFTensor[] { new TFTensor(trainX) }, new TFOutput[] { stacked });
                    Assert.Equal(1000, outputTensors[0].Shape[0]);
                    Assert.Equal(500, outputTensors[0].Shape[1]);
                }
                
            }
        }
    }
}
