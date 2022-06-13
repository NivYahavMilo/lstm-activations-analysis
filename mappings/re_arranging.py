def rearrange_clips(df, where: str = 'columns', with_testretest: bool = False):
    clip_order = ['twomen_clips', 'bridgeville_clips', 'pockets_clips', 'overcome_clips',
                  'inception_clips',
                  'socialnet_clips', 'oceans_clips', 'flower_clips', 'hotel_clips', 'garden_clips', 'dreary_clips',
                  'homealone_clips', 'brokovich_clips', 'starwars_clips',
                  ]

    rest_order = ['twomen_rest_between', 'bridgeville_rest_between',
                  'pockets_rest_between', 'overcome_rest_between', 'inception_rest_between',
                  'socialnet_rest_between', 'oceans_rest_between', 'flower_rest_between', 'hotel_rest_between',
                  'garden_rest_between', 'dreary_rest_between', 'homealone_rest_between', 'brokovich_rest_between',
                  'starwars_rest_between']

    if not with_testretest:
        clip_order[:] = (clip for clip in clip_order if not clip.startswith("test"))
        rest_order[:] = (clip for clip in rest_order if not clip.startswith("test"))

    else:

        clip_order.extend(["testretest1_clips","testretest2_clips","testretest3_clips","testretest4_clips"])
        rest_order.extend(["testretest1_rest_between", "testretest2_rest_between", "testretest3_rest_between", "testretest4_rest_between"])

    clip_order.extend(rest_order)
    if where == 'rows':
        df = df.reindex(clip_order)
    else:
        df = df[clip_order]
    return df
