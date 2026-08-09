# 嵌入式知识体系 · #11 · Makefile 进阶与 CMake 实战

很多嵌入式工程师对构建系统的认知停留在「用 IDE 点一下编译按钮」或者「把别人的 Makefile 复制粘贴改一改」。当项目从单个 `.c` 文件膨胀到几十个目录、上千个文件，再加上交叉编译工具链、条件编译、多个 Flash/Linker 配置时，一个整理好的构建系统直接决定研发效率——甚至决定你下班时间。

这一篇不讲基础变量赋值和 `%` 通配规则，我们直接从**自动化依赖生成**和**现代 CMake 组织方式**入手。

## 一、Makefile 自动依赖生成（`-MMD` / `-MF`）

GCC 的 `-M` 系列选项可以帮 Makefile 自动追踪头文件依赖。这也是嵌入式项目中最常被忽略的关键优化：**没有自动依赖追踪，改了头文件就得 `make clean && make`，或者祈祷增量编译没出问题。**

```makefile
# 典型的 ARM GCC Makefile 自动依赖片段
SRC_DIR   := src
BUILD_DIR := build
SRCS      := $(wildcard $(SRC_DIR)/*.c)
OBJS      := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS      := $(OBJS:.o=.d)          # 依赖文件列表

CFLAGS    := -mcpu=cortex-m4 -mthumb \
             -Wall -Werror -O2 \
             -MMD -MP               # ← 关键：生成 .d 文件
INCLUDES  := -Iinc -Ithird_party/FreeRTOS/include

# -MMD：生成 user 头文件依赖（不包括系统头文件）
# -MP：为每个依赖生成空规则，防止删除头文件后 make 报错

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 自动包含所有 .d 文件（如果存在）
-include $(DEPS)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
```

当 `foo.c` 包含 `foo.h` 和 `bar.h`，GCC 会在第一次编译时生成 `build/foo.d`：

```makefile
# build/foo.d 的内容（自动生成）
build/foo.o: src/foo.c inc/foo.h inc/bar.h
```

下次你修改 `bar.h`，`make` 会检测到 `foo.o` 的依赖变更，重新编译 `foo.c`——**不需要 `make clean`，不需要手动列出头文件**。

**为什么嵌入式项目特别需要这个？** 因为你很可能在改一个全局结构体定义（比如 `osDelay` 的参数类型），而它被 50 个 `.c` 文件包含。没有依赖追踪，即使你记得 `make clean`，也需要浪费一次全量编译（可能在 CI 上跑 10 分钟）。有了 `.d` 文件，只有真正依赖改变的文件才重编译。

## 二、CMake 的 target-based 现代写法

CMake 3.0+ 引入了「target-based」设计哲学。与传统的「变量驱动」不同，现代 CMake 把库、可执行文件、接口视为一等公民，依赖关系通过 target 传播。

```cmake
# CMakeLists.txt —— 现代 target-based 写法
cmake_minimum_required(VERSION 3.16)
project(thermal_printer VERSION 1.0.0 LANGUAGES C ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 可执行目标
add_executable(firmware.elf
    src/main.c
    src/uart.c
    src/timer.c
    startup/startup_stm32f4.s
)

# 第三方库：头文件只导出给链接它的目标
add_library(freertos INTERFACE)
target_include_directories(freertos INTERFACE
    third_party/FreeRTOS/include
    third_party/FreeRTOS/portable/GCC/ARM_CM4F
)

# HAL 库：源文件 + 头文件
add_library(stm32_hal STATIC
    Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal.c
    Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_uart.c
)
target_include_directories(stm32_hal PUBLIC Drivers/STM32F4xx_HAL_Driver/Inc)

# 链接所有依赖
target_link_libraries(firmware.elf PRIVATE
    stm32_hal
    freertos
)

# 全局编译选项（用 generator expressions 控制）
target_compile_options(firmware.elf PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-mcpu=cortex-m4 -mthumb -mfloat-abi=hard>
    $<$<COMPILE_LANGUAGE:ASM>:-mcpu=cortex-m4 -mthumb>
)

target_link_options(firmware.elf PRIVATE
    -mcpu=cortex-m4 -mthumb -mfloat-abi=hard
    -T${CMAKE_SOURCE_DIR}/linker/STM32F407VGTx_FLASH.ld
    -Wl,--gc-sections
)
```

几个关键设计原则：

- **`PRIVATE`**：依赖仅自身使用（不传播）
- **`PUBLIC`**：自身使用且传播给链接者
- **`INTERFACE`**：自身不使用，只传播（适用于纯头文件库如 CMSIS、FreeRTOS 头文件）

这种写法最大的好处是**解耦**：给 `stm32_hal` 加一个 `-DUSE_HAL_DRIVER` 宏，所有链接它的目标自然继承，不需要在每个 `CMakeLists.txt` 里重复声明。

## 三、交叉编译工具链配置

交叉编译的核心是告诉 CMake：「这是一套编译器、一套系统、一套架构，和你的宿主不一样。」

```cmake
# arm-none-eabi-gcc 的 CMake 工具链文件
# 使用时: cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-arm-none-eabi.cmake ..
set(CMAKE_SYSTEM_NAME Generic)       # 不是 Linux/Windows
set(CMAKE_SYSTEM_PROCESSOR arm)

set(TOOLCHAIN_PREFIX arm-none-eabi-)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_AR           ${TOOLCHAIN_PREFIX}ar)
set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}objcopy)
set(CMAKE_SIZE         ${TOOLCHAIN_PREFIX}size)

# 不让 CMake 自作主张运行测试程序（交叉编译下无法运行）
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
```

**常见陷阱**：`set(CMAKE_SYSTEM_NAME Generic)` 很多人会忘记写。如果不写，CMake 默认认为是宿主系统，会尝试检测 `limits.h`、`float.h` 等系统头文件路径——最终编译出来的程序依赖 glibc，放到裸机上直接死机。

## 四、多目录项目的构建组织

当项目模块化到一定程度，`add_subdirectory` 是最好的组织方式：

```
project/
├── CMakeLists.txt          # 根
├── app/
│   ├── CMakeLists.txt      # app 子目录
│   └── main.c
├── drivers/
│   ├── CMakeLists.txt      # 驱动库
│   ├── uart.c / uart.h
│   └── spi.c / spi.h
├── protocols/
│   ├── CMakeLists.txt      # 协议栈
│   └── modbus.c
├── linker/
│   └── stm32.ld
└── cmake/
    └── toolchain-arm-none-eabi.cmake
```

根目录：

```cmake
cmake_minimum_required(VERSION 3.16)
project(firmware)

# 如果目标不是不带系统的裸机，这里建一个 interface 库承载链接脚本等全局信息
add_library(chip INTERFACE)
target_compile_options(chip INTERFACE -mcpu=cortex-m4 -mthumb)
target_link_options(chip INTERFACE -mcpu=cortex-m4 -mthumb -T ${CMAKE_SOURCE_DIR}/linker/stm32.ld)

add_subdirectory(drivers)
add_subdirectory(protocols)
add_subdirectory(app)
```

子目录各管各的依赖：

```cmake
# drivers/CMakeLists.txt
add_library(drivers STATIC uart.c spi.c)
target_link_libraries(drivers PUBLIC chip)
target_include_directories(drivers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
```

```cmake
# app/CMakeLists.txt
add_executable(firmware.elf main.c)
target_link_libraries(firmware.elf PRIVATE drivers protocols)
```

每个子目录的 CMakeLists.txt 只描述自身和直接依赖的关系。当你想把 `drivers` 移植到另一个项目时，只需复制整个目录和它的 CMakeLists.txt——不需要从根目录拆零碎变量。

## 小结

一个设计良好的构建系统，应该让工程师「改了代码只重编必要文件，加了文件只改一个位置」。Makefile 的自动依赖生成 (`-MMD` / `-MP`) 解决了增量编译的可靠性问题；CMake 的 target-based 写法让模块间依赖变得清晰可维护；工具链文件把交叉编译的复杂性收在一个文件里。做好这些基础工作，比研究一个花哨的新 MCU 外设更影响你日常开发的幸福感。

> 🏷️ Makefile CMake 交叉编译 嵌入式构建系统 GCC 自动依赖
