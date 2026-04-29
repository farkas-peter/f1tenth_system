#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <cmath>

using std::placeholders::_1;

class SllidarVisNode : public rclcpp::Node {
public:
    SllidarVisNode() : Node("sllidar_vis_node") {
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::SystemDefaultsQoS(),
            std::bind(&SllidarVisNode::scan_callback, this, _1)
        );

        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/pointcloud", rclcpp::SystemDefaultsQoS()
        );

        RCLCPP_INFO(this->get_logger(), "sllidar_vis_node has been started.");
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = msg->header.stamp;
        cloud.header.frame_id = "lidar_link";

        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");

        std::vector<geometry_msgs::msg::Point> valid_points;
        valid_points.reserve(msg->ranges.size());

        float fov = 180.0;
        float right_limit = fov / 2.0;
        float left_limit = 360.0 - right_limit;
        if (msg->ranges.size() == 720)
        {
            // Express mode
            right_limit = right_limit * 2.0;
            left_limit = 720.0 - right_limit;
        }
        else if (msg->ranges.size() == 1080)
        {
            // Boost mode
            right_limit = right_limit * 3.0;
            left_limit = 1080.0 - right_limit;
        }

        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            if (i <= right_limit || i >= left_limit) {
                float range = msg->ranges[i];
                float angle = msg->angle_min + i * msg->angle_increment;
                geometry_msgs::msg::Point p_map;
                p_map.x = range * std::cos(angle);
                p_map.y = range * std::sin(angle);
                p_map.z = 0.0;

                valid_points.push_back(p_map);
            }
        }

        modifier.resize(valid_points.size());
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

        for (const auto& p : valid_points) {
            *iter_x = p.x;
            *iter_y = p.y;
            *iter_z = p.z;
            ++iter_x;
            ++iter_y;
            ++iter_z;
        }

        pc_pub_->publish(cloud);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SllidarVisNode>());
    rclcpp::shutdown();
    return 0;
}
