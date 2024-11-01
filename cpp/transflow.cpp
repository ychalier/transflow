#include <chrono>
#include <condition_variable>
#include <ctime>
#include <mutex>
#include <signal.h>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>

#define CAP_API cv::CAP_DSHOW
#define RE_OFF 0
#define RE_RANDOM 1
#define RE_LINEAR 2
#define RE_MODE RE_LINEAR
#define MESSAGE_DURATION_MS 500


int bitmap_width = 1280;
int bitmap_height = 720;
int block_size = 8;
int framerate = 24;
double probability = 0;
int remode = RE_OFF;
int flow_width = bitmap_width / block_size;
int flow_height = bitmap_height / block_size;

bool running = true;

int motion_device_id = 0;
cv::VideoCapture motion_capture;
float *mapx_cur, *mapx_buf, *mapy_cur, *mapy_buf, *mapx_base, *mapy_base;
std::mutex motion_mutex;
std::mutex motion_lock_mutex;
std::condition_variable motion_cv;

int bitmap_device_id = 1;
cv::VideoCapture bitmap_capture;
cv::Mat bitmap_frame;
std::mutex bitmap_mutex;
std::mutex bitmap_lock_mutex;
std::condition_variable bitmap_cv;

bool same_device = false;
bool mirror = false;


bool isarg(char* userarg, const char* shortname, const char* longname)
{
    return (strcmp(userarg, shortname) == 0 || strcmp(userarg, longname) == 0);
}


bool parse_args(int argc, char* argv[])
{
    if (argc <= 2)
    {
        return false;
    }
    motion_device_id = std::stoi(argv[1]);
    bitmap_device_id = std::stoi(argv[2]);
    same_device = motion_device_id == bitmap_device_id;
    for (int i = 3; i < argc; i += 2)
    {
        if (isarg(argv[i], "-w", "--width"))
        {
            bitmap_width = std::stoi(argv[i + 1]);
        }
        else if (isarg(argv[i], "-h", "--height"))
        {
            bitmap_height = std::stoi(argv[i + 1]);
        }
        else if (isarg(argv[i], "-b", "--block-size"))
        {
            block_size = std::stoi(argv[i + 1]);
        }
        else if (isarg(argv[i], "-r", "--framerate"))
        {
            framerate = std::stoi(argv[i + 1]);
        }
        else if (isarg(argv[i], "-p", "--probability"))
        {
            probability = std::stod(argv[i + 1]);
        }
        else if (isarg(argv[i], "-m", "--mode"))
        {
            if (strcmp(argv[i + 1], "off") == 0)
            {
                remode = RE_OFF;
            }
            else if (strcmp(argv[i + 1], "random") == 0)
            {
                remode = RE_RANDOM;
            }
            else if (strcmp(argv[i + 1], "linear") == 0)
            {
                remode = RE_LINEAR;
            }
            else
            {
                fprintf(stderr, "Unsupported mode '%s'\n", argv[i]);
                return false;
            }
        }
        else if (isarg(argv[i], "-f", "--flip"))
        {
            mirror = true;
            i--;
        }
        else
        {
            fprintf(stderr, "Unrecognized argument '%s'\n", argv[i]);
            return false;
        }
    }
    if (probability == 0) {
        remode = RE_OFF;
    }
    if (bitmap_width % block_size != 0)
    {
        fprintf(stderr, "Width %d is not divisible by %d\n", bitmap_width, block_size);
        return false;
    }
    if (bitmap_height % block_size != 0)
    {
        fprintf(stderr, "Height %d is not divisible by %d\n", bitmap_height, block_size);
        return false;
    }
    flow_width = bitmap_width / block_size;
    flow_height = bitmap_height / block_size;
    return true;
}


void read_motion_frame()
{
    int x, y, k, zx, zy, dx, dy, x_src, y_src, x_dst, y_dst;
    bool first = true;
    cv::Mat flow(flow_height, flow_width, CV_32FC2);
    cv::Mat frame, prev_grey, grey, frame_full;
    cv::Point2f flow_at;
    std::unique_lock<std::mutex> lock(motion_lock_mutex);
    for (y = 0; y < flow_height; y++)
    {
        for (x = 0; x < flow_width; x++)
        {
            flow.at<cv::Point2f>(y, x) = cv::Point2f(0, 0);
        }
    }
    while (running)
    {
        motion_capture.read(frame_full);
        if (frame_full.empty())
        {
            fprintf(stderr, "Could not grab motion capture frame");
            break;
        }
        cv::resize(frame_full, frame, cv::Size(flow_width, flow_height));
        if (mirror)
        {
            cv::flip(frame, frame, 1);
        }
        if (first)
        {
            first = false;
            cvtColor(frame, prev_grey, cv::COLOR_BGR2GRAY);
            continue;
        }
        else
        {
            cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        }
        calcOpticalFlowFarneback(prev_grey, grey, flow, 0.5, 3, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW);
        prev_grey = grey.clone();
        for (y = 0; y < flow_height; y++)
        {
            for (x = 0; x < flow_width; x++)
            {
                k = y * flow_width + x;
                flow_at = flow.at<cv::Point2f>(y, flow_width - 1 - x);
                dx = (int)std::round(-block_size * flow_at.x);
                dy = (int)std::round(block_size * flow_at.y);
                if (dx != 0 || dy != 0)
                {
                    for (zx = 0; zx < block_size; zx++)
                    {
                        x_src = x * block_size + zx;
                        x_dst = std::max(0, std::min(bitmap_width - 1, x_src + dx));
                        for (zy = 0; zy < block_size; zy++)
                        {
                            y_src = y * block_size + zy;
                            y_dst = std::max(0, std::min(bitmap_height - 1, y_src + dy));
                            mapx_buf[y_dst * bitmap_width + x_dst] = mapx_cur[y_src * bitmap_width + x_src];
                            mapy_buf[y_dst * bitmap_width + x_dst] = mapy_cur[y_src * bitmap_width + x_src];
                        }
                    }
                }
            }
        }
        motion_mutex.lock();
        if (same_device)
        {
            frame_full.copyTo(bitmap_frame);
        }
        for (y = 0; y < bitmap_height; y++)
        {
            for (x = 0; x < bitmap_width; x++)
            {
                k = y * bitmap_width + x;
                switch (remode)
                {
                case RE_OFF:
                    break;
                case RE_RANDOM:
                    if (((double)rand()) / RAND_MAX < probability)
                    {
                        mapx_buf[k] = x;
                        mapy_buf[k] = y;
                    }
                    break;
                case RE_LINEAR:
                    mapx_buf[k] = (1.0f - probability) * mapx_buf[k] + probability * mapx_base[k];
                    mapy_buf[k] = (1.0f - probability) * mapy_buf[k] + probability * mapy_base[k];
                    break;
                default:
                    break;
                }
                mapx_cur[k] = mapx_buf[k];
                mapy_cur[k] = mapy_buf[k];
            }
        }
        motion_mutex.unlock();
        motion_cv.wait(lock);
    }
    flow.release();
    frame.release();
    frame_full.release();
    prev_grey.release();
    grey.release();
    if (same_device)
    {
        bitmap_frame.release();
    }
}


void read_bitmap_frame()
{
    std::unique_lock<std::mutex> lock(bitmap_lock_mutex);
    if (!same_device)
    {
        return;
    }
    while (running)
    {
        bitmap_mutex.lock();
        bitmap_capture.read(bitmap_frame);
        if (bitmap_frame.empty())
        {
            fprintf(stderr, "Could not grab bitmap capture frame");
            break;
        }        
        bitmap_mutex.unlock();
        bitmap_cv.wait(lock);
    }
    bitmap_frame.release();
}


bool open_webcam(cv::VideoCapture *capture, int device_id, int width, int height, int framerate)
{
    capture->open(device_id, CAP_API);
    if (!capture->isOpened())
    {
        fprintf(stderr, "Could not open motion webcam");
        return false;
    }
    capture->set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture->set(cv::CAP_PROP_FRAME_HEIGHT, height);
    capture->set(cv::CAP_PROP_FPS, framerate);
    return true;
}


void unlock_all()
{
    if (!same_device)
    {
        bitmap_mutex.unlock();
        bitmap_cv.notify_all();
    }
    motion_mutex.unlock();
    motion_cv.notify_all();
}


void onsigint(sig_atomic_t s)
{
    running = false;
}


uint64_t time() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}


int main(int argc, char* argv[])
{
    int x, y, k;

    if (!parse_args(argc, argv))
    {
        char prefix[256];
        sprintf(prefix, "usage: %s", argv[0]);
        fprintf(stderr, "%s [-w WIDTH] [-h HEIGHT] [-b BLOCKSIZE] [-r FRAMERATE] [-p PROBABILITY] [-m {off,random,linear}] [-f, --flip]\n%*c <motion device id> <bitmap device id>\n", prefix, strlen(prefix), ' ');
        return EXIT_FAILURE;
    }
    
    // Initialize mapping arrays
    mapx_cur = (float*)malloc(sizeof(float) * bitmap_width * bitmap_height);
    mapy_cur = (float*)malloc(sizeof(float) * bitmap_width * bitmap_height);
    mapx_buf = (float*)malloc(sizeof(float) * bitmap_width * bitmap_height);
    mapy_buf = (float*)malloc(sizeof(float) * bitmap_width * bitmap_height);
    mapx_base = (float*)malloc(sizeof(float) * bitmap_width * bitmap_height);
    mapy_base = (float*)malloc(sizeof(float) * bitmap_width * bitmap_height);
    for (y = 0; y < bitmap_height; y++)
    {
        for (x = 0; x < bitmap_width; x++)
        {
            k = y * bitmap_width + x;
            mapx_cur[k] = x;
            mapy_cur[k] = y;
            mapx_buf[k] = x;
            mapy_buf[k] = y;
            mapx_base[k] = x;
            mapy_base[k] = y;
        }
    }

    // Initialize motion capture
    if (!open_webcam(&motion_capture, motion_device_id, bitmap_width, bitmap_height, framerate))
    {
        return EXIT_FAILURE;
    }
    std::thread motion_reader(read_motion_frame);
    
    if (!same_device)
    {
        // Initialize bitmap catpure
        if (!open_webcam(&bitmap_capture, bitmap_device_id, bitmap_width, bitmap_height, framerate))
        {
            return EXIT_FAILURE;
        }
    }
    std::thread bitmap_reader(read_bitmap_frame);

    // Initialize output window
    bool show_message = false;
    uint64_t message_start;
    char message[48];

    cv::namedWindow("transflow", cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_NORMAL);
    cv::Mat output_image(bitmap_height, bitmap_width, CV_8UC3);

    signal(SIGINT, onsigint);
    while (running)
    {
        motion_mutex.lock();
        if (!same_device)
        {
            bitmap_mutex.lock();
        }
        if (bitmap_frame.rows == 0 || bitmap_frame.cols == 0) {
            unlock_all();
            continue;
        }
        for (y = 0; y < bitmap_height; y++)
        {
            for (x = 0; x < bitmap_width; x++)
            {
                k = y * bitmap_width + x;
                output_image.at<cv::Vec3b>(y, x) = bitmap_frame.at<cv::Vec3b>(std::floor(mapy_cur[k]), std::floor(mapx_cur[k]));
            }
        }
        if (show_message)
        {
            cv::putText(output_image, message, cv::Point(8, 32), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, cv::LINE_8);
            uint64_t now = time();
            if (now - message_start > MESSAGE_DURATION_MS)
            {
                show_message = false;
            }
        }
        imshow("transflow", output_image);
        unlock_all();
        switch(cv::waitKey(1))
        {
            case 27: //ESC
                running = false;
                break;
            case 43: //+
                probability = std::min(1.0, std::max(0.0, probability + 0.01));
                sprintf(message, "%.3f", probability);
                message_start = time();
                show_message = true;
                break;
            case 45: //-
                probability = std::min(1.0, std::max(0.0, probability - 0.01));
                sprintf(message, "%.3f", probability);
                message_start = time();
                show_message = true;
                break;
            case 102: //f
                mirror = !mirror;
                break;
            case 109: //m
                remode = (remode + 1) % 3;
                switch (remode)
                {
                    case RE_OFF:
                        sprintf(message, "OFF");
                        break;
                    case RE_RANDOM:
                        sprintf(message, "RANDOM");
                        break;
                    case RE_LINEAR:
                        sprintf(message, "LINEAR");
                        break;
                }
                message_start = time();
                show_message = true;
                break;
            case 115: //s
                std::vector<int> compression_params;
                char filename[48];
                sprintf(filename, "transflow-%d.jpg", std::time(0));
                cv::imwrite(filename, output_image, compression_params);
                sprintf(message, filename);
                message_start = time();
                show_message = true;
                break;
        }
    }

    running = false;
    cv::destroyAllWindows();
    output_image.release();
    motion_capture.release();
    bitmap_capture.release();
    unlock_all();
    motion_reader.join();
    bitmap_reader.join();
    free(mapx_cur);
    free(mapy_cur);
    free(mapx_buf);
    free(mapy_buf);
    free(mapx_base);
    free(mapy_base);

    return EXIT_SUCCESS;
}