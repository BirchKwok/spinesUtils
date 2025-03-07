import pytest
import time
from spinesUtils.timer import Timer


@pytest.fixture
def timer():
    """提供一个新的Timer实例作为测试fixture"""
    return Timer()


class TestTimer:
    """测试Timer类的功能"""

    def test_init_state(self, timer):
        """测试初始状态"""
        assert not timer.is_started()
        assert not timer.is_running()
        assert not timer.is_paused()
        assert timer.get_elapsed_time() == 0
        assert timer.get_middle_points() == []

    def test_start_end(self, timer):
        """测试基本的开始和结束功能"""
        timer.start()
        assert timer.is_started()
        assert timer.is_running()
        
        # 睡眠一小段时间
        time.sleep(0.1)
        
        elapsed = timer.end()
        assert elapsed >= 0.1
        assert not timer.is_started()

    def test_middle_points(self, timer):
        """测试中间点记录功能"""
        timer.start()
        
        time.sleep(0.1)
        timer.middle_point()
        
        time.sleep(0.1)
        timer.middle_point()
        
        # 应该有两个中间点
        middle_points = timer.get_middle_points()
        assert len(middle_points) == 2
        
        # 第二个点应该比第一个点晚
        assert middle_points[1] > middle_points[0]
        
        # 测试last_timestamp_diff
        time.sleep(0.1)
        last_diff = timer.last_timestamp_diff()
        assert last_diff >= 0.1

    def test_pause_resume(self, timer):
        """测试暂停和恢复功能"""
        timer.start()
        
        time.sleep(0.1)
        timer.pause()
        assert timer.is_paused()
        
        # 暂停期间的时间不应被计入
        paused_time = timer.get_elapsed_time()
        time.sleep(0.2)
        still_paused_time = timer.get_elapsed_time()
        
        # 验证暂停期间时间没有增加
        assert abs(paused_time - still_paused_time) < 0.01
        
        timer.resume()
        assert not timer.is_paused()
        
        time.sleep(0.1)
        final_time = timer.end()
        
        # 总时间应该约为0.2秒（暂停前0.1秒 + 恢复后0.1秒）
        assert final_time >= 0.2
        assert final_time < 0.3  # 允许一点误差，但不应包括暂停的0.2秒

    def test_clear(self, timer):
        """测试清除功能"""
        timer.start()
        time.sleep(0.1)
        timer.middle_point()
        
        timer.clear()
        assert not timer.is_started()
        assert timer.get_middle_points() == []

    def test_sleep(self, timer):
        """测试sleep功能"""
        timer.start()
        
        start_time = time.perf_counter()
        timer.sleep(0.1)
        end_time = time.perf_counter()
        
        # 验证sleep确实暂停了执行
        assert end_time - start_time >= 0.1
        
        # sleep后应该有一个中间点
        assert len(timer.get_middle_points()) == 1
        
        # 测试负数睡眠时间
        with pytest.raises(ValueError):
            timer.sleep(-1)

    def test_session(self, timer):
        """测试会话功能"""
        start_time = time.perf_counter()
        
        with timer.session():
            time.sleep(0.1)
        
        end_time = time.perf_counter()
        
        # 验证会话确实跟踪了时间
        assert end_time - start_time >= 0.1
        assert not timer.is_started()

    def test_middle_point_when_paused(self, timer):
        """测试在暂停状态下记录中间点的行为"""
        timer.start()
        timer.pause()
        
        # 在暂停状态下调用middle_point应返回False
        assert not timer.middle_point()
        
        # 中间点列表应该为空
        assert len(timer.get_middle_points()) == 0

    def test_total_elapsed_time(self, timer):
        """测试total_elapsed_time方法"""
        timer.start()
        time.sleep(0.1)
        
        elapsed1 = timer.total_elapsed_time()
        assert elapsed1 >= 0.1
        
        # 确认total_elapsed_time不会停止计时器
        assert timer.is_running()
        
        time.sleep(0.1)
        elapsed2 = timer.total_elapsed_time()
        
        # 第二次读数应该大于第一次
        assert elapsed2 > elapsed1

    @pytest.mark.parametrize("sleep_duration", [0.01, 0.05, 0.1])
    def test_different_durations(self, timer, sleep_duration):
        """测试不同的时间长度"""
        timer.start()
        time.sleep(sleep_duration)
        elapsed = timer.end()
        assert elapsed >= sleep_duration
        
    @pytest.mark.slow
    def test_longer_duration(self, timer):
        """测试较长时间的计时性能"""
        timer.start()
        time.sleep(0.5)  # 较长时间的测试
        elapsed = timer.end()
        assert elapsed >= 0.5

    @pytest.fixture
    def running_timer(self, timer):
        """提供一个已经启动的计时器"""
        timer.start()
        yield timer
        # 测试结束后清理
        if timer.is_started():
            timer.end()

    def test_with_running_timer(self, running_timer):
        """使用已启动的计时器测试"""
        assert running_timer.is_running()
        time.sleep(0.1)
        assert running_timer.get_elapsed_time() >= 0.1
        
    def test_timer_precision(self, timer_precision):
        """测试计时器精度，使用会话级别的fixture"""
        print(f"\n计时器精度: {timer_precision:.9f}秒")
        timer = Timer()
        timer.start()
        # 验证计时器精度足够我们的测试需求
        assert timer.get_elapsed_time() < 0.01  # 初始时间应该非常小
        
    @pytest.mark.parametrize("chain_length", [1, 3, 5])
    def test_method_chaining(self, timer, chain_length):
        """测试方法链式调用"""
        # 构建一个方法链，考虑到middle_point可能返回布尔值的情况
        timer.start()
        for _ in range(chain_length):
            time.sleep(0.01)
            result = timer.middle_point()
            assert result is True or result is timer  # middle_point应该返回True或self
            
        # 验证链式调用结果
        assert len(timer.get_middle_points()) == chain_length
        assert timer.is_running()
        
    def test_multiple_pause_resume(self, timer):
        """测试多次暂停和恢复"""
        timer.start()
        
        time.sleep(0.1)
        timer.pause()
        time.sleep(0.1)  # 这段时间不应该被计入
        timer.resume()
        
        time.sleep(0.1)
        timer.pause()
        time.sleep(0.1)  # 这段时间不应该被计入
        timer.resume()
        
        time.sleep(0.1)
        
        total = timer.end()
        # 总时间应该约为0.3秒（3个0.1秒的运行段）
        assert 0.3 <= total < 0.35


@pytest.mark.slow
class TestTimerAdvanced:
    """更高级的计时器测试，需要更长时间运行"""
    
    def test_stability(self, timer):
        """测试多个时间段的稳定性"""
        results = []
        
        for _ in range(3):
            timer.start()
            time.sleep(0.1)
            results.append(timer.end())
            
        # 每次结果应该相近
        assert max(results) - min(results) < 0.05 