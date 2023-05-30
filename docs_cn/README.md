LlamaIndex文档

## 文档贡献者指南

`docs`目录包含LlamaIndex文档的sphinx源文本，访问https://gpt-index.readthedocs.io/可以阅读完整的文档。

本指南适用于有兴趣在本地运行LlamaIndex文档，对其进行修改并做出贡献的任何人。LlamaIndex是由其背后的活跃社区创建的，您总是可以为该项目和文档做出贡献。

## 构建文档

如果您还没有，请将LlamaIndex Github存储库克隆到本地目录：

```bash
git clone https://github.com/jerryjliu/llama_index.git && cd llama_index
```

安装构建文档所需的所有依赖项（主要是`sphinx`及其扩展）：

```bash
pip install -r docs/requirements.txt
```

构建sphinx文档：

```bash
cd docs
make html
```

现在文档HTML文件已经生成在`docs/_build/html`目录下，您可以使用以下命令在本地预览它：

```bash
python -m http.server 8000 -d _build/html
```

然后在浏览器中打开http://0.0.0.0:8000/查看生成的文档。

##### 查看文档

我们建议在开发过程中使用sphinx-autobuild，它提供了一个实时重新加载服务器，当保存更改时，它会重新构建文档并自动刷新任何打开的页面。这可以缩短反馈循环，有助于提高书写文档时的生产力。

只需从LlamaIndex项目的根目录运行以下命令：
```bash
make watch-docs
```