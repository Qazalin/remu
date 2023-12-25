
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_9_6_3_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s8, s13, 0x2aaaaaab                           // 000000001714: 9708FF0D 2AAAAAAB
	s_mul_i32 s2, s15, 0x4d                                    // 00000000171C: 9602FF0F 0000004D
	s_lshr_b32 s9, s8, 31                                      // 000000001724: 85099F08
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_add_i32 s9, s8, s9                                       // 00000000172C: 81090908
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_mul_i32 s8, s9, 6                                        // 000000001734: 96088609
	s_mul_i32 s10, s9, 7                                       // 000000001738: 960A8709
	s_sub_i32 s12, s13, s8                                     // 00000000173C: 818C080D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001744: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001748: 82070307
	s_ashr_i32 s11, s10, 31                                    // 00000000174C: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001754: 8482820A
	s_add_u32 s9, s6, s2                                       // 000000001758: 80090206
	s_addc_u32 s3, s7, s3                                      // 00000000175C: 82030307
	s_ashr_i32 s13, s12, 31                                    // 000000001760: 860D9F0C
	s_mul_i32 s2, s14, 6                                       // 000000001764: 9602860E
	s_lshl_b64 s[6:7], s[12:13], 2                             // 000000001768: 8486820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704B9
	s_add_u32 s10, s9, s6                                      // 000000001770: 800A0609
	s_addc_u32 s11, s3, s7                                     // 000000001774: 820B0703
	s_ashr_i32 s3, s2, 31                                      // 000000001778: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000177C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s12, s0, s2                                      // 000000001784: 800C0200
	s_addc_u32 s13, s1, s3                                     // 000000001788: 820D0301
	s_load_b128 s[0:3], s[12:13], null                         // 00000000178C: F4080006 F8000000
	s_clause 0x2                                               // 000000001794: BF850002
	s_load_b64 s[16:17], s[10:11], null                        // 000000001798: F4040405 F8000000
	s_load_b64 s[18:19], s[10:11], 0x1c                        // 0000000017A0: F4040485 F800001C
	s_load_b64 s[10:11], s[10:11], 0x38                        // 0000000017A8: F4040285 F8000038
	s_load_b64 s[12:13], s[12:13], 0x10                        // 0000000017B0: F4040306 F8000010
	s_waitcnt lgkmcnt(0)                                       // 0000000017B8: BF89FC07
	v_fma_f32 v0, s16, s0, 0                                   // 0000000017BC: D6130000 02000010
	s_mul_i32 s0, s15, 0x144                                   // 0000000017C4: 9600FF0F 00000144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017CC: BF8704A1
	v_fmac_f32_e64 v0, s17, s1                                 // 0000000017D0: D52B0000 00000211
	s_ashr_i32 s1, s0, 31                                      // 0000000017D8: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017DC: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017E0: BF8700C1
	v_fmac_f32_e64 v0, s18, s2                                 // 0000000017E4: D52B0000 00000412
	s_mul_i32 s2, s14, 54                                      // 0000000017EC: 9602B60E
	s_add_u32 s4, s4, s0                                       // 0000000017F0: 80040004
	s_addc_u32 s5, s5, s1                                      // 0000000017F4: 82050105
	v_fmac_f32_e64 v0, s19, s3                                 // 0000000017F8: D52B0000 00000613
	s_ashr_i32 s3, s2, 31                                      // 000000001800: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001804: BF870099
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001808: 84808202
	v_fmac_f32_e64 v0, s10, s12                                // 00000000180C: D52B0000 0000180A
	s_add_u32 s2, s4, s0                                       // 000000001814: 80020004
	s_addc_u32 s3, s5, s1                                      // 000000001818: 82030105
	s_ashr_i32 s9, s8, 31                                      // 00000000181C: 86099F08
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001820: BF870001
	v_fmac_f32_e64 v0, s11, s13                                // 000000001824: D52B0000 00001A0B
	s_lshl_b64 s[0:1], s[8:9], 2                               // 00000000182C: 84808208
	v_mov_b32_e32 v1, 0                                        // 000000001830: 7E020280
	s_add_u32 s0, s2, s0                                       // 000000001834: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001838: 82010103
	v_max_f32_e32 v0, 0, v0                                    // 00000000183C: 20000080
	s_add_u32 s0, s0, s6                                       // 000000001840: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001844: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 000000001848: DC6A0000 00000001
	s_nop 0                                                    // 000000001850: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001854: BFB60003
	s_endpgm                                                   // 000000001858: BFB00000
