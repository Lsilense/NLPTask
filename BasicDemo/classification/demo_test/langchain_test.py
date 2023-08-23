import os
os.environ["OPENAI_API_KEY"] = "..."
 
from typing import Any, List, Mapping, Optional

#导入所需的类和接口
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

#定义一个名为CustomLLM的子类，继承自LLM类
class CustomLLM(LLM):
    
    # 类的成员变量，类型为整型
    n: int

    # 用于指定该子类对象的类型
    @property
    def _llm_type(self) -> str:
        return "custom"

    # 重写基类方法，根据用户输入的prompt来响应用户，返回字符串
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    # 返回一个字典类型，包含LLM的唯一标识
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    
llm = CustomLLM(n=50)
print(llm("This is a foobar thing"))