using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text; // [추가됨] StringBuilder 사용
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
        
        private readonly string[] _classNames = new string[]
         { "overripe", "breaking - stage","un-healthy", "ripe", "unripe", "half-riping-stage" };

        private readonly Dictionary<string, string> _translationMap = new Dictionary<string, string>
        {
            { "breaking - stage", "익어가는 중" },
            { "half-riping-stage", "반숙" },
            { "overripe", "과숙 (지나치게 익음)" },
            { "ripe", "익음 (정상)" },
            { "un-healthy", "비정상 (병든 망고)" },
            { "unripe", "안 익음 (미숙)" }
        };

        // ----------------------------------------------------------------------
        // [추가됨 1] 님의 요청: "누적 분류 통계"를 저장할 변수 (Dictionary)
        // ----------------------------------------------------------------------
        private Dictionary<string, int> _cumulativeStats;


        private const int ModelInputSize = 224;

        public MainWindow()
        {
            InitializeComponent();
            
            // ----------------------------------------------------------------------
            // [추가됨 2] 누적 통계 데이터(Dictionary)를 0으로 초기화
            // ----------------------------------------------------------------------
            InitializeCumulativeStats();
            UpdateStatsDisplay(); // 화면에 "전부 0"인 초기 상태 표시
            
            // (기존 코드)
            LoadOnnxModel();
            LoadFarmDashboardData(); // [추가됨] 이전 단계의 대시보드 데이터 로드
        }

        // [추가됨] 이전 단계에서 빠진 '시뮬레이션 데이터' 로드 메서드
        private void LoadFarmDashboardData()
        {
            FarmEnvTextBlock.Text = "온도: 24.5°C\n습도: 65.2 %\nCO2: 450 ppm";
            WeatherTextBlock.Text = "맑음 / 25°C\n풍속: 3 m/s (NW)\n강수 확률: 10%";
            SeasonInfoTextBlock.Text = "망고 주 수확철 (8주차)\n시장 가격: 15,000원/kg (↑)";
        }

        // ----------------------------------------------------------------------
        // [추가됨 3] 누적 통계 Dictionary를 초기화하는 메서드
        // ----------------------------------------------------------------------
        private void InitializeCumulativeStats()
        {
            _cumulativeStats = new Dictionary<string, int>();
            
            // '번역 사전'에 있는 모든 "한글 이름"을 키로 사용하여
            // 누적 통계 딕셔너리를 0으로 세팅합니다.
            foreach (var koreanName in _translationMap.Values)
            {
                if (!_cumulativeStats.ContainsKey(koreanName))
                {
                    _cumulativeStats.Add(koreanName, 0);
                }
            }
        }

        // ----------------------------------------------------------------------
        // [추가됨 4] 누적 통계 딕셔너리의 데이터를 UI(TextBlock)에 표시하는 메서드
        // ----------------------------------------------------------------------
        private void UpdateStatsDisplay()
        {
            // StringBuilder를 사용해 통계 문자열을 효율적으로 만듭니다.
            StringBuilder statsBuilder = new StringBuilder();

            foreach (var entry in _cumulativeStats)
            {
                // (예: "익음 (정상): 5 개")
                statsBuilder.AppendLine($"{entry.Key}: {entry.Value} 개");
            }

            // XAML에 있는 TextBlock의 내용을 업데이트합니다.
            CumulativeStatsTextBlock.Text = statsBuilder.ToString();
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

                    var (predictedClass, confidence, allScores) = await RunPredictionAsync(imagePath);

                    ResultTextBlock.Text = $"{predictedClass}";
                    ConfidenceTextBlock.Text = $"{confidence * 100:F2} %";
                    FullResultsListView.ItemsSource = allScores.OrderByDescending(s => s.Confidence);

                    // ----------------------------------------------------------------------
                    // [수정됨 5] 님의 요청: AI가 예측한 후, 누적 통계를 업데이트합니다.
                    // ----------------------------------------------------------------------
                    if (_cumulativeStats.ContainsKey(predictedClass))
                    {
                        _cumulativeStats[predictedClass]++; // (예: "익음 (정상)" 카운트 1 증가)
                    }
                    UpdateStatsDisplay(); // 화면의 통계표를 즉시 새로고침
                    // ----------------------------------------------------------------------
                }
                catch (Exception ex)
                {
                    ResultTextBlock.Text = "예측 오류";
                    ConfidenceTextBlock.Text = "---";
                    MessageBox.Show($"이미지 처리 또는 예측 중 오류 발생: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

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
                            string englishName = _classNames[i]; 
                            string koreanName = _translationMap[englishName];

                            allScores.Add(new PredictionScore
                            {
                                ClassName = koreanName,
                                Confidence = probabilities[i]
                            });
                        }
                        
                        float maxConfidence = probabilities.Max();
                        int maxIndex = Array.IndexOf(probabilities, maxConfidence);
                        string englishTopClass = _classNames[maxIndex];
                        string koreanTopClass = _translationMap[englishTopClass];

                        return (koreanTopClass, maxConfidence, allScores);
                    }
                }
            });
        }
    }
}
