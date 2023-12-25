
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_45_65>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mul_i32 s6, s15, 0x41                                    // 000000001608: 9606FF0F 00000041
	v_mov_b32_e32 v0, 0xff800000                               // 000000001610: 7E0002FF FF800000
	s_ashr_i32 s7, s6, 31                                      // 000000001618: 86079F06
	s_mov_b32 s4, s15                                          // 00000000161C: BE84000F
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001620: 84868206
	s_waitcnt lgkmcnt(0)                                       // 000000001624: BF89FC07
	s_add_u32 s5, s2, s6                                       // 000000001628: 80050602
	s_addc_u32 s6, s3, s7                                      // 00000000162C: 82060703
	s_mov_b64 s[2:3], 0                                        // 000000001630: BE820180
	s_set_inst_prefetch_distance 0x1                           // 000000001634: BF840001
	s_nop 0                                                    // 000000001638: BF800000
	s_nop 0                                                    // 00000000163C: BF800000
	s_add_u32 s20, s5, s2                                      // 000000001640: 80140205
	s_addc_u32 s21, s6, s3                                     // 000000001644: 82150306
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001648: BF870001
	v_max_f32_e32 v0, v0, v0                                   // 00000000164C: 20000100
	s_clause 0x2                                               // 000000001650: BF850002
	s_load_b256 s[8:15], s[20:21], null                        // 000000001654: F40C020A F8000000
	s_load_b128 s[16:19], s[20:21], 0x20                       // 00000000165C: F408040A F8000020
	s_load_b32 s7, s[20:21], 0x30                              // 000000001664: F40001CA F8000030
	s_add_u32 s2, s2, 52                                       // 00000000166C: 8002B402
	s_addc_u32 s3, s3, 0                                       // 000000001670: 82038003
	s_cmpk_eq_i32 s2, 0x104                                    // 000000001674: B1820104
	s_waitcnt lgkmcnt(0)                                       // 000000001678: BF89FC07
	v_max_f32_e64 v1, s8, s8                                   // 00000000167C: D5100001 00001008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001684: BF870091
	v_max_f32_e32 v0, v1, v0                                   // 000000001688: 20000101
	v_max3_f32 v0, s10, s9, v0                                 // 00000000168C: D61C0000 0400120A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001694: BF870091
	v_max3_f32 v0, s12, s11, v0                                // 000000001698: D61C0000 0400160C
	v_max3_f32 v0, s14, s13, v0                                // 0000000016A0: D61C0000 04001A0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A8: BF870091
	v_max3_f32 v0, s16, s15, v0                                // 0000000016AC: D61C0000 04001E10
	v_max3_f32 v0, s18, s17, v0                                // 0000000016B4: D61C0000 04002212
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016BC: BF870001
	v_max3_f32 v0, s7, s19, v0                                 // 0000000016C0: D61C0000 04002607
	s_cbranch_scc0 65501                                       // 0000000016C8: BFA1FFDD <r_45_65+0x40>
	s_set_inst_prefetch_distance 0x2                           // 0000000016CC: BF840002
	s_ashr_i32 s5, s4, 31                                      // 0000000016D0: 86059F04
	v_mov_b32_e32 v1, 0                                        // 0000000016D4: 7E020280
	s_lshl_b64 s[2:3], s[4:5], 2                               // 0000000016D8: 84828204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000016DC: BF870009
	s_add_u32 s0, s0, s2                                       // 0000000016E0: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000016E4: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000016E8: DC6A0000 00000001
	s_nop 0                                                    // 0000000016F0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016F4: BFB60003
	s_endpgm                                                   // 0000000016F8: BFB00000
