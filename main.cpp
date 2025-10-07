#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <windows.h>
#include <fstream>
#include <random> // ��������֧��

// ���������ɫ
cv::Scalar generate_random_color(int seed) {
    static std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(50, 255);
    return cv::Scalar(dist(rng), dist(rng), dist(rng));
}

// ����COCO������Ʋ�������ɫ
std::vector<std::pair<std::string, cv::Scalar>> load_coco_names_with_colors(const std::string& file_path) {
    std::vector<std::pair<std::string, cv::Scalar>> class_info;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open COCO names file: " << file_path << std::endl;
        return class_info;
    }

    std::string line;
    int class_id = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Ϊÿ��������ɹ̶������ɫ
            cv::Scalar color = generate_random_color(class_id);
            class_info.push_back(std::make_pair(line, color));
        }
        class_id++;
    }

    file.close();
    return class_info;
}

// ����ͼ��ߴ粢���ֿ�߱�
cv::Mat resize_with_aspect_ratio(const cv::Mat& image, int size, float& ratio, int& pad_w, int& pad_h) {
    int orig_width = image.cols;
    int orig_height = image.rows;

    ratio = min(static_cast<float>(size) / orig_width,
        static_cast<float>(size) / orig_height);

    int new_width = static_cast<int>(orig_width * ratio);
    int new_height = static_cast<int>(orig_height * ratio);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    pad_w = (size - new_width) / 2;
    pad_h = (size - new_height) / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_h, pad_h, pad_w, pad_w,
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return padded;
}

// ���Ƽ��������������ƺ���ɫ��
void draw_detections(cv::Mat& image,
    const std::vector<int64_t>& labels,
    const std::vector<float>& boxes,
    const std::vector<float>& scores,
    const std::vector<std::pair<std::string, cv::Scalar>>& class_info, // �޸�Ϊ����ɫ�������Ϣ
    float ratio, int pad_w, int pad_h,
    float thrh = 0.4) {
    size_t num_detections = scores.size();
    for (size_t i = 0; i < num_detections; ++i) {
        if (scores[i] <= thrh) continue;

        // �����߽������
        float x0 = (boxes[4 * i] - pad_w) / ratio;
        float y0 = (boxes[4 * i + 1] - pad_h) / ratio;
        float x1 = (boxes[4 * i + 2] - pad_w) / ratio;
        float y1 = (boxes[4 * i + 3] - pad_h) / ratio;

        // ȷ��������ͼ��Χ��
        x0 = max(0.0f, min(x0, static_cast<float>(image.cols - 1)));
        y0 = max(0.0f, min(y0, static_cast<float>(image.rows - 1)));
        x1 = max(0.0f, min(x1, static_cast<float>(image.cols - 1)));
        y1 = max(0.0f, min(y1, static_cast<float>(image.rows - 1)));

        // ��ȡ�����Ϣ
        std::string label = "Unknown";
        cv::Scalar color(0, 0, 255); // Ĭ�Ϻ�ɫ
        if (labels[i] >= 0 && static_cast<size_t>(labels[i]) < class_info.size()) {
            label = class_info[static_cast<size_t>(labels[i])].first;
            color = class_info[static_cast<size_t>(labels[i])].second;
        }

        // ���Ʊ߽��ʹ������ض���ɫ��
        cv::rectangle(image, cv::Point(static_cast<int>(x0), static_cast<int>(y0)),
            cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
            color, 2);

        // ������Ŷ�
        std::string label_text = cv::format("%s: %.2f", label.c_str(), scores[i]);

        // ���Ʊ�ǩ������ʹ����ͬ��ɫ����ǳ��
        cv::Scalar text_bg_color = color * 0.7;
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image,
            cv::Point(static_cast<int>(x0), static_cast<int>(y0) - text_size.height - 5),
            cv::Point(static_cast<int>(x0) + text_size.width, static_cast<int>(y0)),
            text_bg_color, cv::FILLED);

        // ����������ƺ����Ŷȣ���ɫ���֣�
        cv::putText(image, label_text,
            cv::Point(static_cast<int>(x0), static_cast<int>(y0) - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ����ͼ��
void process_image(Ort::Session& session,
    const cv::Mat& image,
    int size,
    const std::vector<std::pair<std::string, cv::Scalar>>& class_info) { // �޸Ĳ�������
    float ratio;
    int pad_w, pad_h;
    cv::Mat resized = resize_with_aspect_ratio(image, size, ratio, pad_w, pad_h);

    // ׼����������
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // ��ͼ���HWCת��ΪCHW��ʽ
    cv::Mat chw_image;
    cv::dnn::blobFromImage(float_img, chw_image);

    // ��ȡ����Ԫ������
    size_t data_size = chw_image.total() * chw_image.channels();
    std::vector<float> input_data(data_size);
    std::memcpy(input_data.data(), chw_image.ptr<float>(), data_size * sizeof(float));

    // ��������tensor
    std::vector<int64_t> input_shape = { 1, 3, size, size };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), data_size, input_shape.data(), input_shape.size()
        );

    // ����orig_target_sizes tensor
    std::vector<int64_t> orig_target_sizes = { static_cast<int64_t>(resized.rows), static_cast<int64_t>(resized.cols) };
    std::vector<int64_t> target_shape = { 1, 2 };
    Ort::Value target_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, orig_target_sizes.data(), orig_target_sizes.size(),
        target_shape.data(), target_shape.size()
        );

    // ��������
    std::vector<const char*> input_names = { "images", "orig_target_sizes" };
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    inputs.push_back(std::move(target_tensor));

    // �������
    const char* output_names[] = { "labels", "boxes", "scores" };
    auto outputs = session.Run(Ort::RunOptions{ nullptr },
        input_names.data(), inputs.data(), inputs.size(),
        output_names, 3);

    // �������
    auto& labels_out = outputs[0];
    auto& boxes_out = outputs[1];
    auto& scores_out = outputs[2];

    const int64_t* labels_ptr = labels_out.GetTensorData<int64_t>();
    const float* boxes_ptr = boxes_out.GetTensorData<float>();
    const float* scores_ptr = scores_out.GetTensorData<float>();

    // ��ȡ�������
    auto labels_shape = labels_out.GetTensorTypeAndShapeInfo().GetShape();
    size_t num_detections = labels_shape[1];

    // �����������
    std::vector<int64_t> labels(labels_ptr, labels_ptr + num_detections);
    std::vector<float> boxes(boxes_ptr, boxes_ptr + num_detections * 4);
    std::vector<float> scores(scores_ptr, scores_ptr + num_detections);

    // ���Ʋ������������������Ϣ��
    cv::Mat result_image = image.clone();
    draw_detections(result_image, labels, boxes, scores, class_info, ratio, pad_w, pad_h);

    // Windows ·������
    std::string result_path = "onnx_result.jpg";
    //cv::imwrite(result_path, result_image);
    //std::cout << "Image processing complete. Result saved as '" << result_path << "'." << std::endl;
}

// ������Ƶ
void process_video(Ort::Session& session,
    const std::string& video_path,
    int size,
    const std::vector<std::pair<std::string, cv::Scalar>>& class_info) { // �޸Ĳ�������
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return;
    }

    // ��ȡ��Ƶ����
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Windows ��Ƶ������
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    // ������Ƶд����
    std::string output_path = "onnx_result.mp4";
    cv::VideoWriter writer(output_path, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error creating video writer for: " << output_path << std::endl;
        return;
    }

    size_t frame_count = 0;
    std::cout << "Processing video frames..." << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        float ratio;
        int pad_w, pad_h;
        cv::Mat resized = resize_with_aspect_ratio(frame, size, ratio, pad_w, pad_h);

        // Ԥ����
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // ��ͼ���HWCת��ΪCHW��ʽ
        cv::Mat chw_image;
        cv::dnn::blobFromImage(float_img, chw_image);

        // ��ȡ����Ԫ������
        size_t data_size = chw_image.total() * chw_image.channels();
        std::vector<float> input_data(data_size);
        std::memcpy(input_data.data(), chw_image.ptr<float>(), data_size * sizeof(float));

        std::vector<int64_t> input_shape = { 1, 3, size, size };
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), data_size, input_shape.data(), input_shape.size()
            );

        std::vector<int64_t> orig_target_sizes = { static_cast<int64_t>(resized.rows), static_cast<int64_t>(resized.cols) };
        std::vector<int64_t> target_shape = { 1, 2 };
        Ort::Value target_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, orig_target_sizes.data(), orig_target_sizes.size(),
            target_shape.data(), target_shape.size()
            );

        // ����
        std::vector<const char*> input_names = { "images", "orig_target_sizes" };
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        inputs.push_back(std::move(target_tensor));

        const char* output_names[] = { "labels", "boxes", "scores" };
        auto outputs = session.Run(Ort::RunOptions{ nullptr },
            input_names.data(), inputs.data(), inputs.size(),
            output_names, 3);

        // �������
        auto& labels_out = outputs[0];
        auto& boxes_out = outputs[1];
        auto& scores_out = outputs[2];

        const int64_t* labels_ptr = labels_out.GetTensorData<int64_t>();
        const float* boxes_ptr = boxes_out.GetTensorData<float>();
        const float* scores_ptr = scores_out.GetTensorData<float>();

        auto labels_shape = labels_out.GetTensorTypeAndShapeInfo().GetShape();
        size_t num_detections = labels_shape[1];

        std::vector<int64_t> labels(labels_ptr, labels_ptr + num_detections);
        std::vector<float> boxes(boxes_ptr, boxes_ptr + num_detections * 4);
        std::vector<float> scores(scores_ptr, scores_ptr + num_detections);

        // ���ƽ�������������Ϣ��
        cv::Mat result_frame = frame.clone();
        draw_detections(result_frame, labels, boxes, scores, class_info, ratio, pad_w, pad_h);
        writer.write(result_frame);

        if (++frame_count % 10 == 0) {
            std::cout << "Processed " << frame_count << " frames..." << std::endl;
        }
    }

    cap.release();
    writer.release();
    std::cout << "Video processing complete. Result saved as '" << output_path << "'." << std::endl;
}

int main(int argc, char* argv[]) {
    // Windows ����̨����
   /* SetConsoleOutputCP(CP_UTF8);
    setvbuf(stdout, nullptr, _IOFBF, 1000);

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " --onnx <onnx_model> --input <input_file> --names <coco_names_file>" << std::endl;
        return 1;
    }*/

    std::string onnx_path = "deimv2_dinov3_m_coco.onnx";// argv[1];
    std::string input_path = "bus.jpg";// argv[2];
    std::string coco_names_path = "coco.names";// argv[3];

    // ����COCO������Ʋ�������ɫ
    std::vector<std::pair<std::string, cv::Scalar>> class_info = load_coco_names_with_colors(coco_names_path);
    if (class_info.empty()) {
        std::cerr << "Warning: No class names loaded. Using numeric labels with default colors." << std::endl;
    }
    else {
        std::cout << "Loaded " << class_info.size() << " class names from " << coco_names_path << std::endl;
        // ��ʾǰ5��������ɫʾ��
        std::cout << "Color examples for first 5 classes:" << std::endl;
        for (int i = 0; i < min(5, static_cast<int>(class_info.size())); ++i) {
            cv::Scalar color = class_info[i].second;
            std::cout << "  " << class_info[i].first << ": [B:" << static_cast<int>(color[0])
                << ", G:" << static_cast<int>(color[1])
                << ", R:" << static_cast<int>(color[2]) << "]" << std::endl;
        }
    }

    try {
        // ��ʼ��ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DEIMv2");
        Ort::SessionOptions session_options;

        // Windows �Ż�����
        //session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        //����CUDA����
         //OrtCUDAProviderOptions cudaOption;
         //session_options.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider

         // Load the ONNX model into the session
#ifdef _WIN32
        std::wstring w_modelPath(onnx_path.begin(), onnx_path.end());
        Ort::Session session(env, w_modelPath.c_str(), session_options);
#else
        Ort::Session session(env, onnx_path.c_str(), session_options);
#endif

        // ��ȡ����ߴ�
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = input_tensor_info.GetShape();
        int size = static_cast<int>(input_shape[2]);

        while (1)
        {
            double start, end;
            start = clock();

            // ����������ͣ�ͼ�����Ƶ��
            cv::Mat image = cv::imread(input_path);
            if (!image.empty()) {
                process_image(session, image, size, class_info);
            }
            else {
                // ������Ϊ��Ƶ��
                cv::VideoCapture test_cap(input_path);
                if (test_cap.isOpened()) {
                    test_cap.release();
                    process_video(session, input_path, size, class_info);
                }
                else {
                    std::cerr << "Error: Input file is not a valid image or video: " << input_path << std::endl;
                    return 1;
                }
            }
            end = clock();
            printf("process cost:%.2f-ms\n", end - start);
        }
        
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "General Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
