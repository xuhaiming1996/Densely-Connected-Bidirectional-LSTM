'''
给定一个字符串 s 和一些长度相同的单词 words。在 s 中找出可以恰好串联 words 中所有单词的子串的起始位置。

注意子串要与 words 中的单词完全匹配，中间不能有其他字符，但不需要考虑 words 中单词串联的顺序。

示例 1:

输入:
  s = "barfoothefoobarman",
  words = ["foo","bar"]
输出: [0,9]
解释: 从索引 0 和 9 开始的子串分别是 "barfoor" 和 "foobar" 。
输出的顺序不重要, [9,0] 也是有效答案。
示例 2:

输入:
  s = "wordgoodstudentgoodword",
  words = ["word","student"]
输出: []

'''


class Solution:
    @staticmethod
    def findSubstring(s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        if s=="" or len(words)==0:
            return []
        res_index=[]
        word_num = len(words)
        word_len = len(words[0])
        words_len=word_num*word_len
        s_len=len(s)
        index=0
        words_new=sorted(words)
        words_str="".join(words_new)

        while index+words_len <= s_len:
            index_word = s[index:word_len+index]

            if index_word in words:
                tmp_str = s[index:words_len + index]

                tmp_list = [tmp_str[i:i + word_len] for i in range(0, len(tmp_str), word_len)]
                if "".join(sorted(tmp_list))==words_str:
                    res_index.append(index)
            index+=1

        return res_index


if __name__=="__main__":
    # s = "barfoothefoobarman"
    # words = ["foo", "bar"]

    s="ababaab"
    words=["ab", "ba", "ba"]


    s="acaaa"
    words=["aa", "ca"]
    print(Solution.findSubstring(s,words))













