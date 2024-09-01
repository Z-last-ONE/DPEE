import json

# 读取JSON文件
with open('./data/fewFC/dev.json', 'r',encoding='utf-8') as file:
    train_data = [json.loads(x) for x in file.readlines()]  # 使用json.load()方法将文件内容解析为Python字典
newdict = {"eventTypeList": ['质押', '股份股权转让', '投资', '减持', '起诉', '收购', '判决', '签署合同', '担保', '中标'],
           "roleList": ['number', 'obj-per', 'collateral', 'date', 'sub-per', 'obj-org', 'proportion', 'sub-org',
                        'money', 'target-company', 'sub', 'obj', 'share-per', 'share-org', 'title', 'way',
                        'institution', 'amount'],
           "eventTypeMeaning": ['质押是指公司或个人将所持有的资产（如股权、股票）作为抵押品',
                                '股份股权转让是指公司股东将其所持有的部分或全部股份或者股权转让给其他股东或第三方',
                                '投资是指公司将资金投入到其他公司、项目或金融工具中，以期获得经济回报或实现战略目标',
                                '减持是指公司现有股东出售其所持有的部分或全部股份，减少在公司的持股比例',
                                '起诉是指公司或个人因某种纠纷向法院提起诉讼，寻求法律救济或解决争端',
                                '收购是指一家公司购买另一家公司的股份或者资产，以取得其控制权的行为',
                                '判决是指法院对某一法律案件作出的正式裁决，决定案件的法律责任和后续行动',
                                '签署合同是指两个或多个主体之间正式签订合同、协议、仪式等',
                                '担保是指公司为其他公司或个人提供财务担保，承诺在被担保方无法履行债务时代其偿还',
                                '中标是指公司在投标活动中被选为最终中标方，获得合同或项目的执行权'],
           "roleMeaning": ['number是指事件中涉及的数量信息或者数值，如股份数量、交易数量等',
                           'obj-per是指事件的目标对象是个人，如股东、投资者等',
                           'collateral是指用作担保或抵押的资产，如股票、股权、股份等',
                           'date是指事件发生的日期或时间信息',
                           'sub-per是指事件的主体是个人',
                           'obj-org是指事件的目标对象是组织或公司',
                           'proportion是指事件中涉及的比例信息',
                           'sub-org是指事件的主体是组织或公司',
                           'money是指涉及的资金或金额信息',
                           'target-company是指股份股权转让事件中涉及的目标公司',
                           'sub是指事件的主体，既可以是个人也可以是组织',
                           'obj是指事件的目标对象，既可以是个人也可以是组织',
                           'share-per是指减持股份占持股人所持股份的比例',
                           'share-org是指减持股份占公司总股本的比例',
                           'title是指减持事件中涉及的职位或头衔信息，如股东、董事等',
                           'way是指事件的具体执行方式或手段',
                           'institution是指做出判决的机构，通常是法院或者仲裁机构',
                           'amount是指事件涉及的数量或金额']}
data=[]
for i in train_data:
    bigDict={}
    bigDict['id']=i['id']
    newdict['text']=i['content']
    bigDict['prompt']=json.dumps(newdict,ensure_ascii=False)
    # key=i['id']
    data.append(bigDict)
with open('devInputPrompt.json', 'w', encoding='utf-8') as file:
    for i,item in enumerate(data):
        json_line = json.dumps(item, ensure_ascii=False)  # 将字典转换为JSON字符串
        file.write(json_line + '\n')

