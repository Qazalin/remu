
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_11_3_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s8, s13, 0x55555556                           // 000000001714: 9708FF0D 55555556
	s_mul_i32 s2, s15, 0x4d                                    // 00000000171C: 9602FF0F 0000004D
	s_lshr_b32 s9, s8, 31                                      // 000000001724: 85099F08
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_add_i32 s8, s8, s9                                       // 00000000172C: 81080908
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_mul_i32 s12, s8, 3                                       // 000000001734: 960C8308
	s_mul_i32 s8, s8, 7                                        // 000000001738: 96088708
	s_sub_i32 s10, s13, s12                                    // 00000000173C: 818A0C0D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001744: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001748: 82070307
	s_ashr_i32 s9, s8, 31                                      // 00000000174C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001754: 84828208
	s_add_u32 s8, s6, s2                                       // 000000001758: 80080206
	s_addc_u32 s3, s7, s3                                      // 00000000175C: 82030307
	s_ashr_i32 s11, s10, 31                                    // 000000001760: 860B9F0A
	s_mul_i32 s2, s14, 5                                       // 000000001764: 9602850E
	s_lshl_b64 s[6:7], s[10:11], 2                             // 000000001768: 8486820A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704B9
	s_add_u32 s16, s8, s6                                      // 000000001770: 80100608
	s_addc_u32 s17, s3, s7                                     // 000000001774: 82110703
	s_ashr_i32 s3, s2, 31                                      // 000000001778: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000177C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s18, s0, s2                                      // 000000001784: 80120200
	s_addc_u32 s19, s1, s3                                     // 000000001788: 82130301
	s_load_b128 s[0:3], s[18:19], null                         // 00000000178C: F4080009 F8000000
	s_clause 0x1                                               // 000000001794: BF850001
	s_load_b128 s[8:11], s[16:17], null                        // 000000001798: F4080208 F8000000
	s_load_b32 s13, s[16:17], 0x10                             // 0000000017A0: F4000348 F8000010
	s_load_b32 s16, s[18:19], 0x10                             // 0000000017A8: F4000409 F8000010
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_fma_f32 v0, s8, s0, 0                                    // 0000000017B4: D6130000 02000008
	s_mul_i32 s0, s15, 0xc6                                    // 0000000017BC: 9600FF0F 000000C6
	s_mul_i32 s8, s14, 33                                      // 0000000017C4: 9608A10E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017C8: BF8704A1
	v_fmac_f32_e64 v0, s9, s1                                  // 0000000017CC: D52B0000 00000209
	s_ashr_i32 s1, s0, 31                                      // 0000000017D4: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017D8: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017DC: BF8700C1
	v_fmac_f32_e64 v0, s10, s2                                 // 0000000017E0: D52B0000 0000040A
	s_add_u32 s2, s4, s0                                       // 0000000017E8: 80020004
	s_addc_u32 s4, s5, s1                                      // 0000000017EC: 82040105
	s_ashr_i32 s9, s8, 31                                      // 0000000017F0: 86099F08
	v_fmac_f32_e64 v0, s11, s3                                 // 0000000017F4: D52B0000 0000060B
	s_lshl_b64 s[0:1], s[8:9], 2                               // 0000000017FC: 84808208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001800: BF8700A9
	s_add_u32 s2, s2, s0                                       // 000000001804: 80020002
	s_addc_u32 s3, s4, s1                                      // 000000001808: 82030104
	v_fmac_f32_e64 v0, s13, s16                                // 00000000180C: D52B0000 0000200D
	s_ashr_i32 s13, s12, 31                                    // 000000001814: 860D9F0C
	v_mov_b32_e32 v1, 0                                        // 000000001818: 7E020280
	s_lshl_b64 s[0:1], s[12:13], 2                             // 00000000181C: 8480820C
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001820: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001824: 20000080
	s_add_u32 s0, s2, s0                                       // 000000001828: 80000002
	s_addc_u32 s1, s3, s1                                      // 00000000182C: 82010103
	s_add_u32 s0, s0, s6                                       // 000000001830: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001834: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 000000001838: DC6A0000 00000001
	s_nop 0                                                    // 000000001840: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001844: BFB60003
	s_endpgm                                                   // 000000001848: BFB00000
