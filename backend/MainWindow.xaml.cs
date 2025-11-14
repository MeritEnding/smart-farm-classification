using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
// ONNX 런타임 라이브러리
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
// 이미지 처리를 위한 SixLabors 라이브러리
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MangoClassifierWPF
{
    public partial class MainWindow : Window
    {
        private InferenceSession? _session;

        // ----------------------------------------------------------------------
        // ⚠️ [중요] 님께서 학습시킨 클래스 이름으로 이 배열을 수정하세요!
        // ----------------------------------------------------------------------
        private readonly string[] _classNames = new string[] { "overripe", "breaking - stage","un-healthy", "ripe", "unripe", "half-riping-stage" }; // ⬅️⬅️⬅️ 예시입니다. 꼭 수정하세요!

        private const int ModelInputSize = 224;

        public MainWindow()
        {
            InitializeComponent();
            LoadOnnxModel();
        }

        private void LoadOnnxModel()
        {
            try
            {
                string modelPath = System.IO.Path.Combine(AppContext.BaseDirectory, "best.onnx");

                if (!File.Exists(modelPath))
                {
                    MessageBox.Show($"모델 파일을 찾을 수 없습니다: {modelPath}\nbest.onnx 파일을 프로젝트에 추가하고 '속성'에서 '출력 디렉터리로 복사'를 '새 버전이면 복사'로 설정했는지 확인하세요.",
                        "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                var sessionOptions = new SessionOptions();
                sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;

                _session = new InferenceSession(modelPath, sessionOptions);

                ResultTextBlock.Text = "모델 로드 성공.";
                ConfidenceTextBlock.Text = "이미지를 선택하세요.";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"모델 로드 중 심각한 오류 발생: {ex.Message}", "모델 로드 실패", MessageBoxButton.OK, MessageBoxImage.Error);
                Application.Current.Shutdown();
            }
        }

        private async void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            if (_session == null)
            {
                MessageBox.Show("모델이 아직 로드되지 않았습니다.", "오류", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|모든 파일 (*.*)|*.json",
                Title = "테스트할 이미지 선택"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                string imagePath = openFileDialog.FileName;

                try
                {
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.EndInit();
                    SourceImage.Source = bitmap;

                    ResultTextBlock.Text = "예측 중...";
                    ConfidenceTextBlock.Text = "...";

                    var (predictedClass, confidence) = await RunPredictionAsync(imagePath);

                    ResultTextBlock.Text = $"{predictedClass}";
                    ConfidenceTextBlock.Text = $"{confidence * 100:F2} %";
                }
                catch (Exception ex)
                {
                    ResultTextBlock.Text = "예측 오류";
                    ConfidenceTextBlock.Text = "---";
                    MessageBox.Show($"이미지 처리 또는 예측 중 오류 발생: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private async System.Threading.Tasks.Task<(string, float)> RunPredictionAsync(string imagePath)
        {
            return await System.Threading.Tasks.Task.Run(() =>
            {
                using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
                {
                    image.Mutate(x =>
                        x.Resize(ModelInputSize, ModelInputSize)
                    );

                    var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });

                    for (int y = 0; y < image.Height; y++)
                    {
                        for (int x = 0; x < image.Width; x++)
                        {
                            var pixel = image[x, y];
                            tensor[0, 0, y, x] = pixel.R / 255.0f;
                            tensor[0, 1, y, x] = pixel.G / 255.0f;
                            tensor[0, 2, y, x] = pixel.B / 255.0f;
                        }
                    }

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("images", tensor)
                    };

                    if (_session == null)
                    {
                        throw new InvalidOperationException("ONNX 세션이 초기화되지 않았습니다.");
                    }

                    using (var results = _session.Run(inputs))
                    {
                        // [수정됨] 5. 결과 후처리
                        var output = results.First().AsTensor<float>();

                        // [수정] Softmax 함수를 호출하지 않습니다.
                        // output.ToArray() 자체가 이미 확률 배열입니다.
                        var probabilities = output.ToArray();

                        // 가장 높은 확률을 가진 클래스 찾기
                        float maxConfidence = probabilities.Max();
                        int maxIndex = Array.IndexOf(probabilities, maxConfidence);
                        string predictedClass = _classNames[maxIndex];

                        return (predictedClass, maxConfidence);
                    }
                }
            });
        }

        // [수정됨] Softmax 함수를 완전히 삭제했습니다.
    }
}
