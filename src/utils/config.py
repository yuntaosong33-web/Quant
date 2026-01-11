"""
配置和日志工具模块

提供日志系统配置和YAML配置文件加载/保存功能。
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
import sys

import yaml


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    配置日志系统
    
    Parameters
    ----------
    level : int, optional
        日志级别，默认为 INFO
    log_file : Optional[str]
        日志文件路径，如果为None则只输出到控制台
    format_string : Optional[str]
        日志格式字符串
    
    Returns
    -------
    logging.Logger
        配置好的根日志器
    
    Examples
    --------
    >>> logger = setup_logging(level=logging.DEBUG)
    >>> logger.info("系统启动")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        )
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Parameters
    ----------
    config_path : Union[str, Path]
        配置文件路径
    
    Returns
    -------
    Dict[str, Any]
        配置字典
    
    Raises
    ------
    FileNotFoundError
        当配置文件不存在时
    yaml.YAMLError
        当YAML解析失败时
    
    Examples
    --------
    >>> config = load_config("config/strategy_config.yaml")
    >>> print(config["strategy"]["name"])
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logging.info(f"配置文件加载成功: {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    保存配置到YAML文件
    
    Parameters
    ----------
    config : Dict[str, Any]
        配置字典
    config_path : Union[str, Path]
        保存路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    logging.info(f"配置文件保存成功: {config_path}")

