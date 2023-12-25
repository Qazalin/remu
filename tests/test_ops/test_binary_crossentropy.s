
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_320>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000160C: F4040200 F8000010
	v_mov_b32_e32 v0, 0                                        // 000000001614: 7E000280
	s_mov_b64 s[10:11], 0                                      // 000000001618: BE8A0180
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s0, s6, s10                                      // 000000001620: 80000A06
	s_addc_u32 s1, s7, s11                                     // 000000001624: 82010B07
	s_add_u32 s2, s8, s10                                      // 000000001628: 80020A08
	s_addc_u32 s3, s9, s11                                     // 00000000162C: 82030B09
	s_add_u32 s10, s10, 8                                      // 000000001630: 800A880A
	s_load_b64 s[2:3], s[2:3], null                            // 000000001634: F4040081 F8000000
	s_load_b64 s[0:1], s[0:1], null                            // 00000000163C: F4040000 F8000000
	s_addc_u32 s11, s11, 0                                     // 000000001644: 820B800B
	s_cmpk_eq_i32 s10, 0x500                                   // 000000001648: B18A0500
	s_waitcnt lgkmcnt(0)                                       // 00000000164C: BF89FC07
	v_mul_f32_e64 v3, 0xbfb8aa3b, s2                           // 000000001650: D5080003 000004FF BFB8AA3B
	v_mul_f32_e64 v1, s0, 0.5                                  // 00000000165C: D5080001 0001E000
	v_mul_f32_e64 v2, s1, 0.5                                  // 000000001664: D5080002 0001E001
	v_cmp_gt_f32_e64 s12, s1, 0                                // 00000000166C: D414000C 00010001
	v_cmp_gt_f32_e64 s13, s0, 0                                // 000000001674: D414000D 00010000
	v_mul_f32_e64 v4, 0xbfb8aa3b, s3                           // 00000000167C: D5080004 000006FF BFB8AA3B
	v_cmp_nlt_f32_e64 vcc_lo, s0, 0                            // 000000001688: D41E006A 00010000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001690: BF870214
	v_cndmask_b32_e64 v2, v2, s1, s12                          // 000000001694: D5010002 00300302
	v_cndmask_b32_e64 v1, v1, s0, s13                          // 00000000169C: D5010001 00340101
	v_cmp_nlt_f32_e64 s0, s1, 0                                // 0000000016A4: D41E0000 00010001
	v_cmp_gt_f32_e64 s1, 0xc2fc0000, v3                        // 0000000016AC: D4140001 000206FF C2FC0000
	v_cmp_gt_f32_e64 s2, 0xc2fc0000, v4                        // 0000000016B8: D4140002 000208FF C2FC0000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000016C4: BF870193
	v_cndmask_b32_e64 v2, 0, v2, s0                            // 0000000016C8: D5010002 00020480
	v_cndmask_b32_e64 v7, 0, 0x42800000, s1                    // 0000000016D0: D5010007 0005FE80 42800000
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 0000000016DC: 02020280
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000016E0: BF870224
	v_cndmask_b32_e64 v8, 0, 0x42800000, s2                    // 0000000016E4: D5010008 0009FE80 42800000
	v_cndmask_b32_e64 v5, 1.0, 0x1f800000, s1                  // 0000000016F0: D5010005 0005FEF2 1F800000
	v_dual_sub_f32 v10, -1.0, v2 :: v_dual_add_f32 v3, v3, v7  // 0000000016FC: C94804F3 0A020F03
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001704: BF8701A3
	v_dual_sub_f32 v9, -1.0, v1 :: v_dual_add_f32 v4, v4, v8   // 000000001708: C94802F3 09041104
	v_cmp_gt_f32_e32 vcc_lo, 1.0, v2                           // 000000001710: 7C2804F2
	v_mul_f32_e32 v8, 0.5, v10                                 // 000000001714: 101014F0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001718: BF870194
	v_exp_f32_e32 v3, v3                                       // 00000000171C: 7E064B03
	v_mul_f32_e32 v7, 0.5, v9                                  // 000000001720: 100E12F0
	v_cmp_gt_f32_e64 s0, 1.0, v1                               // 000000001724: D4140000 000202F2
	v_exp_f32_e32 v4, v4                                       // 00000000172C: 7E084B04
	v_cndmask_b32_e64 v6, 1.0, 0x1f800000, s2                  // 000000001730: D5010006 0009FEF2 1F800000
	v_cndmask_b32_e64 v8, v8, -v2, vcc_lo                      // 00000000173C: D5010008 41AA0508
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001744: BF8700C3
	v_cndmask_b32_e64 v7, v7, -v1, s0                          // 000000001748: D5010007 40020307
	v_cmp_nlt_f32_e64 s0, 1.0, v2                              // 000000001750: D41E0000 000204F2
	s_waitcnt_depctr 0xfff                                     // 000000001758: BF880FFF
	v_dual_mul_f32 v2, v5, v3 :: v_dual_mul_f32 v3, v6, v4     // 00000000175C: C8C60705 02020906
	v_add_f32_e32 v2, 1.0, v2                                  // 000000001764: 060404F2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001768: BF870121
	v_div_scale_f32 v4, null, v2, v2, 1.0                      // 00000000176C: D6FC7C04 03CA0502
	v_div_scale_f32 v10, vcc_lo, 1.0, v2, 1.0                  // 000000001774: D6FC6A0A 03CA04F2
	v_rcp_f32_e32 v6, v4                                       // 00000000177C: 7E0C5504
	s_waitcnt_depctr 0xfff                                     // 000000001780: BF880FFF
	v_fma_f32 v12, -v4, v6, 1.0                                // 000000001784: D613000C 23CA0D04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000178C: BF870091
	v_dual_add_f32 v3, 1.0, v3 :: v_dual_fmac_f32 v6, v12, v6  // 000000001790: C90006F2 03060D0C
	v_div_scale_f32 v5, null, v3, v3, 1.0                      // 000000001798: D6FC7C05 03CA0703
	v_div_scale_f32 v11, s1, 1.0, v3, 1.0                      // 0000000017A0: D6FC010B 03CA06F2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000017A8: BF870193
	v_mul_f32_e32 v12, v10, v6                                 // 0000000017AC: 10180D0A
	v_rcp_f32_e32 v9, v5                                       // 0000000017B0: 7E125505
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B4: BF870091
	v_fma_f32 v14, -v4, v12, v10                               // 0000000017B8: D613000E 242A1904
	v_fmac_f32_e32 v12, v14, v6                                // 0000000017C0: 56180D0E
	s_waitcnt_depctr 0xfff                                     // 0000000017C4: BF880FFF
	v_fma_f32 v13, -v5, v9, 1.0                                // 0000000017C8: D613000D 23CA1305
	v_fma_f32 v4, -v4, v12, v10                                // 0000000017D0: D6130004 242A1904
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017D8: BF870112
	v_fmac_f32_e32 v9, v13, v9                                 // 0000000017DC: 5612130D
	v_div_fmas_f32 v4, v4, v6, v12                             // 0000000017E0: D6370004 04320D04
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000017E8: BF870122
	v_mul_f32_e32 v13, v11, v9                                 // 0000000017EC: 101A130B
	s_mov_b32 vcc_lo, s1                                       // 0000000017F0: BEEA0001
	v_div_fixup_f32 v2, v4, v2, 1.0                            // 0000000017F4: D6270002 03CA0504
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017FC: BF870092
	v_fma_f32 v15, -v5, v13, v11                               // 000000001800: D613000F 242E1B05
	v_dual_sub_f32 v4, 1.0, v2 :: v_dual_fmac_f32 v13, v15, v9 // 000000001808: C94004F2 040C130F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001810: BF870111
	v_cmp_class_f32_e64 s1, v4, 0x90                           // 000000001814: D47E0001 0001FF04 00000090
	v_fma_f32 v5, -v5, v13, v11                                // 000000001820: D6130005 242E1B05
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001828: BF870121
	v_div_fmas_f32 v5, v5, v9, v13                             // 00000000182C: D6370005 04361305
	v_cmp_nlt_f32_e32 vcc_lo, 1.0, v1                          // 000000001834: 7C3C02F2
	v_div_fixup_f32 v3, v5, v3, 1.0                            // 000000001838: D6270003 03CA0705
	v_cndmask_b32_e64 v5, -1.0, v8, s0                         // 000000001840: D5010005 000210F3
	v_cmp_class_f32_e64 s0, v2, 0x90                           // 000000001848: D47E0000 0001FF02 00000090
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001854: BF870113
	v_cmp_class_f32_e64 s2, v3, 0x90                           // 000000001858: D47E0002 0001FF03 00000090
	v_cndmask_b32_e64 v8, 1.0, 0x4f800000, s0                  // 000000001864: D5010008 0001FEF2 4F800000
	v_cndmask_b32_e32 v1, -1.0, v7, vcc_lo                     // 000000001870: 02020EF3
	v_cndmask_b32_e64 v7, 1.0, 0x4f800000, s1                  // 000000001874: D5010007 0005FEF2 4F800000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001880: BF870214
	v_cndmask_b32_e64 v10, 0, 0x42000000, s2                   // 000000001884: D501000A 0009FE80 42000000
	v_mul_f32_e32 v2, v2, v8                                   // 000000001890: 10041102
	v_sub_f32_e32 v6, 1.0, v3                                  // 000000001894: 080C06F2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001898: BF870144
	v_mul_f32_e32 v4, v4, v7                                   // 00000000189C: 10080F04
	v_cndmask_b32_e64 v7, 1.0, 0x4f800000, s2                  // 0000000018A0: D5010007 0009FEF2 4F800000
	v_cndmask_b32_e64 v8, 0, 0x42000000, s0                    // 0000000018AC: D5010008 0001FE80 42000000
	v_log_f32_e32 v2, v2                                       // 0000000018B8: 7E044F02
	v_mul_f32_e32 v3, v3, v7                                   // 0000000018BC: 10060F03
	v_cndmask_b32_e64 v7, 0, 0x42000000, s1                    // 0000000018C0: D5010007 0005FE80 42000000
	s_waitcnt_depctr 0xfff                                     // 0000000018CC: BF880FFF
	v_sub_f32_e32 v2, v2, v8                                   // 0000000018D0: 08041102
	v_log_f32_e32 v3, v3                                       // 0000000018D4: 7E064F03
	v_add_f32_e32 v8, 1.0, v5                                  // 0000000018D8: 06100AF2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000018DC: BF8700C2
	v_mul_f32_e32 v2, 0x3f317218, v2                           // 0000000018E0: 100404FF 3F317218
	s_waitcnt_depctr 0xfff                                     // 0000000018E8: BF880FFF
	v_sub_f32_e32 v3, v3, v10                                  // 0000000018EC: 08061503
	v_log_f32_e32 v4, v4                                       // 0000000018F0: 7E084F04
	v_mul_f32_e32 v3, 0x3f317218, v3                           // 0000000018F4: 100606FF 3F317218
	s_waitcnt_depctr 0xfff                                     // 0000000018FC: BF880FFF
	v_dual_sub_f32 v4, v4, v7 :: v_dual_add_f32 v7, 1.0, v1    // 000000001900: C9480F04 040602F2
	v_cmp_class_f32_e64 s3, v6, 0x90                           // 000000001908: D47E0003 0001FF06 00000090
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001914: BF870112
	v_mul_f32_e32 v4, 0x3f317218, v4                           // 000000001918: 100808FF 3F317218
	v_cndmask_b32_e64 v9, 1.0, 0x4f800000, s3                  // 000000001920: D5010009 000DFEF2 4F800000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000192C: BF870112
	v_mul_f32_e32 v4, v7, v4                                   // 000000001930: 10080907
	v_mul_f32_e32 v6, v6, v9                                   // 000000001934: 100C1306
	v_cndmask_b32_e64 v9, 0, 0x42000000, s3                    // 000000001938: D5010009 000DFE80 42000000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001944: BF870193
	v_fma_f32 v1, v1, v2, -v4                                  // 000000001948: D6130001 84120501
	v_log_f32_e32 v6, v6                                       // 000000001950: 7E0C4F06
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001954: BF8700B1
	v_add_f32_e32 v0, v0, v1                                   // 000000001958: 06000300
	s_waitcnt_depctr 0xfff                                     // 00000000195C: BF880FFF
	v_sub_f32_e32 v6, v6, v9                                   // 000000001960: 080C1306
	v_mul_f32_e32 v6, 0x3f317218, v6                           // 000000001964: 100C0CFF 3F317218
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000196C: BF870091
	v_mul_f32_e32 v6, v8, v6                                   // 000000001970: 100C0D08
	v_fma_f32 v2, v5, v3, -v6                                  // 000000001974: D6130002 841A0705
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000197C: BF870001
	v_add_f32_e32 v0, v0, v2                                   // 000000001980: 06000500
	s_cbranch_scc0 65317                                       // 000000001984: BFA1FF25 <r_320+0x1c>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001988: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v0, 0x3b4ccccd, v0  // 00000000198C: CA060080 010000FF 3B4CCCCD
	global_store_b32 v1, v0, s[4:5]                            // 000000001998: DC6A0000 00040001
	s_nop 0                                                    // 0000000019A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000019A4: BFB60003
	s_endpgm                                                   // 0000000019A8: BFB00000
