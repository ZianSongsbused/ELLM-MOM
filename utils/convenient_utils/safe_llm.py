from utils.convenient_utils.suppress_useless_print import suppress_everything


# 不做客户端超时，只做500重试
def safe_llm_invoke_retry_500(llm, messages, *, tag="", max_attempts=2):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        print(f"[LLM-START] tag={tag}, attempt={attempt}")
        try:
            # with suppress_everything():
            resp = llm.invoke(messages)  # 阻塞等待
            # print(resp)   # 只有content=xxxx没有其他字段
            print(f"[LLM-END]   tag={tag}, attempt={attempt}")
            return resp

        except Exception as e:
            status = getattr(e, "status_code", None)
            print(f"[LLM-ERROR] tag={tag}, attempt={attempt}, status={status}")

            # 只有500才重试，其它直接抛出
            if status == 500:
                last_exc = e
                if attempt < max_attempts:
                    print(f"[LLM-RETRY] tag={tag}, attempt={attempt}, status=500 -> retry")
                    continue
            raise  # 非500 or 最后一次失败

    raise last_exc

