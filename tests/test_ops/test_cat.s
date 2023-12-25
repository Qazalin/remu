
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_45_195_9>:
	s_load_b256 s[0:7], s[0:1], null                           // 000000001700: F40C0000 F8000000
	s_mul_i32 s10, s14, 9                                      // 000000001708: 960A890E
	s_mul_i32 s9, s15, 0x249                                   // 00000000170C: 9609FF0F 00000249
	s_add_i32 s11, s10, s13                                    // 000000001714: 810B0D0A
	s_mov_b32 s8, s13                                          // 000000001718: BE88000D
	s_add_i32 s12, s11, s9                                     // 00000000171C: 810C090B
	s_mov_b32 s9, 0                                            // 000000001720: BE890080
	s_cmp_gt_i32 s14, 64                                       // 000000001724: BF02C00E
	s_mov_b32 s16, 0                                           // 000000001728: BE900080
	s_cbranch_scc0 43                                          // 00000000172C: BFA1002B <E_45_195_9+0xdc>
	s_waitcnt lgkmcnt(0)                                       // 000000001730: BF89FC07
	s_add_i32 s2, s14, 0xffffffbf                              // 000000001734: 8102FF0E FFFFFFBF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000173C: BF870009
	s_cmp_gt_u32 s2, 64                                        // 000000001740: BF08C002
	s_cbranch_scc0 50                                          // 000000001744: BFA10032 <E_45_195_9+0x110>
	s_cmpk_lt_i32 s14, 0x82                                    // 000000001748: B38E0082
	s_mov_b32 s2, 0                                            // 00000000174C: BE820080
	s_cbranch_scc1 7                                           // 000000001750: BFA20007 <E_45_195_9+0x70>
	s_ashr_i32 s13, s12, 31                                    // 000000001754: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001758: BF870499
	s_lshl_b64 s[2:3], s[12:13], 2                             // 00000000175C: 8482820C
	s_add_u32 s2, s6, s2                                       // 000000001760: 80020206
	s_addc_u32 s3, s7, s3                                      // 000000001764: 82030307
	s_load_b32 s2, s[2:3], -0x1248                             // 000000001768: F4000081 F81FEDB8
	s_mul_i32 s4, s15, 0x6db                                   // 000000001770: 9604FF0F 000006DB
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	v_add_f32_e64 v0, s16, s9                                  // 00000000177C: D5030000 00001210
	s_ashr_i32 s5, s4, 31                                      // 000000001784: 86059F04
	v_mov_b32_e32 v1, 0                                        // 000000001788: 7E020280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000178C: 84848204
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001790: BF8704C2
	v_add_f32_e32 v0, s2, v0                                   // 000000001794: 06000002
	s_add_u32 s3, s0, s4                                       // 000000001798: 80030400
	s_addc_u32 s4, s1, s5                                      // 00000000179C: 82040501
	s_ashr_i32 s11, s10, 31                                    // 0000000017A0: 860B9F0A
	s_lshl_b64 s[0:1], s[10:11], 2                             // 0000000017A4: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017A8: BF8704B9
	s_add_u32 s3, s3, s0                                       // 0000000017AC: 80030003
	s_addc_u32 s4, s4, s1                                      // 0000000017B0: 82040104
	s_ashr_i32 s9, s8, 31                                      // 0000000017B4: 86099F08
	s_lshl_b64 s[0:1], s[8:9], 2                               // 0000000017B8: 84808208
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017BC: BF870009
	s_add_u32 s0, s3, s0                                       // 0000000017C0: 80000003
	s_addc_u32 s1, s4, s1                                      // 0000000017C4: 82010104
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C8: DC6A0000 00000001
	s_nop 0                                                    // 0000000017D0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D4: BFB60003
	s_endpgm                                                   // 0000000017D8: BFB00000
	s_ashr_i32 s13, s12, 31                                    // 0000000017DC: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017E0: BF870009
	s_lshl_b64 s[16:17], s[12:13], 2                           // 0000000017E4: 8490820C
	s_waitcnt lgkmcnt(0)                                       // 0000000017E8: BF89FC07
	s_add_u32 s2, s2, s16                                      // 0000000017EC: 80021002
	s_addc_u32 s3, s3, s17                                     // 0000000017F0: 82031103
	s_load_b32 s16, s[2:3], null                               // 0000000017F4: F4000401 F8000000
	s_add_i32 s2, s14, 0xffffffbf                              // 0000000017FC: 8102FF0E FFFFFFBF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001804: BF870009
	s_cmp_gt_u32 s2, 64                                        // 000000001808: BF08C002
	s_cbranch_scc1 65486                                       // 00000000180C: BFA2FFCE <E_45_195_9+0x48>
	s_ashr_i32 s13, s12, 31                                    // 000000001810: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001814: BF870499
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000001818: 8482820C
	s_add_u32 s2, s4, s2                                       // 00000000181C: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001820: 82030305
	s_load_b32 s9, s[2:3], -0x924                              // 000000001824: F4000241 F81FF6DC
	s_cmpk_lt_i32 s14, 0x82                                    // 00000000182C: B38E0082
	s_mov_b32 s2, 0                                            // 000000001830: BE820080
	s_cbranch_scc0 65479                                       // 000000001834: BFA1FFC7 <E_45_195_9+0x54>
	s_branch 65485                                             // 000000001838: BFA0FFCD <E_45_195_9+0x70>
