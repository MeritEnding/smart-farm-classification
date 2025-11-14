using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MangoClassifierWPF
{
    public class PredictionScore
    {
        public string ClassName { get; set; } = "";
        public double Confidence { get; set; }
    }


    public partial class MainWindow : Window
    {
        private InferenceSession? _session;

        // ----------------------------------------------------------------------
        // [수정됨 1] 🚨 모델이 학습한 "알파벳 순서"와 100% 일치시켰습니다.
        // ----------------------------------------------------------------------
        private readonly string[] _classNames = new string[]
         { "overripe", "breaking - stage","un-healthy", "ripe", "unripe", "half-riping-stage" };

        // ----------------------------------------------------------------------
        // [추가됨 2] 🇰🇷 영어 클래스 이름을 한글로 번역하기 위한 "번역 사전"
        // (이곳에서 원하시는 한글 이름으로 수정하실 수 있습니다.)
        // ----------------------------------------------------------------------
        private readonly Dictionary<string, string> _translationMap = new Dictionary<string, string>
        {
            { "breaking - stage", "익어가는 중" },
            { "half-riping-stage", "반숙" },
            { "overripe", "과숙 (지나치게 익음)" },
            { "ripe", "익음 (정상)" },
            { "un-healthy", "비정상 (병든 망고)" },
            { "unripe", "안 익음 (미숙)" }
        };

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
                    MessageBox.Show($"모델 파일을 찾을 수 없습니다: {modelPath}", "모델 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error);
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
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|모든 파일 (*.*)|*.*",
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
                    FullResultsListView.ItemsSource = null;

                    // (predictedClass, confidence, allScores) 값은
                    // 이제 "한글로 번역된" 결과가 담겨서 옵니다.
                    var (predictedClass, confidence, allScores) = await RunPredictionAsync(imagePath);

                    ResultTextBlock.Text = $"{predictedClass}";
                    ConfidenceTextBlock.Text = $"{confidence * 100:F2} %";
                    FullResultsListView.ItemsSource = allScores.OrderByDescending(s => s.Confidence);
                }
                catch (Exception ex)
                {
                    ResultTextBlock.Text = "예측 오류";
                    ConfidenceTextBlock.Text = "---";
                    MessageBox.Show($"이미지 처리 또는 예측 중 오류 발생: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        // 반환 타입 (string TopClass, float TopConfidence, List<PredictionScore> AllScores)
        // 여기서 string TopClass는 이제 "한글" 이름이 됩니다.
        private async System.Threading.Tasks.Task<(string TopClass, float TopConfidence, List<PredictionScore> AllScores)> RunPredictionAsync(string imagePath)
        {
            return await System.Threading.Tasks.Task.Run(() =>
            {
                using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath))
                {
                    image.Mutate(x =>
                        x.Resize(new ResizeOptions
                        {
                            Size = new SixLabors.ImageSharp.Size(ModelInputSize, ModelInputSize),
                            Mode = SixLabors.ImageSharp.Processing.ResizeMode.Crop
                        })
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

                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

                    if (_session == null)
                    {
                        throw new InvalidOperationException("ONNX 세션이 초기화되지 않았습니다.");
                    }

                    using (var results = _session.Run(inputs))
                    {
                        var output = results.First().AsTensor<float>();
                        var probabilities = output.ToArray();

                        var allScores = new List<PredictionScore>();
                        for (int i = 0; i < probabilities.Length; i++)
                        {
                            // ----------------------------------------------------------
                            // [수정됨 3] 영어 이름을 한글로 번역
                            // ----------------------------------------------------------
                            string englishName = _classNames[i]; // (예: "ripe")
                            string koreanName = _translationMap[englishName]; // (예: "익음 (정상)")

                            allScores.Add(new PredictionScore
                            {
                                ClassName = koreanName, // <-- 한글 이름 저장
                                Confidence = probabilities[i]
                            });
                        }

                        // 10. 가장 높은 점수 찾기 (기존 로직)
                        float maxConfidence = probabilities.Max();
                        int maxIndex = Array.IndexOf(probabilities, maxConfidence);

                        // ----------------------------------------------------------
                        // [수정됨 4] Top 클래스도 한글로 번역
                        // ----------------------------------------------------------
                        string englishTopClass = _classNames[maxIndex]; // (예: "ripe")
                        string koreanTopClass = _translationMap[englishTopClass]; // (예: "익음 (정상)")

                        // 11. "한글로 번역된" Top 클래스 이름과 전체 리스트를 반환
                        return (koreanTopClass, maxConfidence, allScores);
                    }
                }
            });
        }
    }
}
